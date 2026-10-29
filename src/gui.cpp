// Win32 GUI: click a piece, then click its destination. The engine thinks on a worker thread.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "board.h"
#include "search.h"
#include "tablebase.h"

namespace {

enum {
    WM_ENGINE_DONE = WM_APP + 1,
    WM_REFRESH = WM_APP + 2,
};
enum {
    ID_NEW_WHITE = 101, ID_NEW_BLACK, ID_UNDO, ID_FLIP, ID_COPY_FEN, ID_PASTE_FEN, ID_EXIT,
    ID_SWITCH_SIDES = 201, ID_MOVE_NOW,
    ID_TIME_BASE = 300,  // + index into THINK_TIMES
};
const int THINK_TIMES[] = {1000, 3000, 5000, 10000, 30000};

const COLORREF C_BG = RGB(38, 36, 33);
const COLORREF C_PANEL_TEXT = RGB(222, 220, 214);
const COLORREF C_DIM_TEXT = RGB(150, 146, 140);
const COLORREF C_ACCENT = RGB(129, 182, 76);
const COLORREF C_LIGHT = RGB(240, 217, 181);
const COLORREF C_DARK = RGB(181, 136, 99);
const COLORREF C_LAST_LIGHT = RGB(205, 210, 106);
const COLORREF C_LAST_DARK = RGB(170, 162, 58);
const COLORREF C_SELECT = RGB(130, 151, 105);
const COLORREF C_CHECK = RGB(226, 82, 70);
const COLORREF C_HINT = RGB(90, 110, 60);

struct App {
    HWND hwnd = nullptr;
    int dpi = 96;
    Board board;
    std::vector<Move> played;
    std::vector<std::string> san;
    Color startStm = WHITE;
    int startFullmove = 1;
    Color human = WHITE;
    bool flipped = false;
    int selected = -1;
    std::vector<Move> targets;
    int thinkIndex = 1;
    std::string gameOver;

    std::thread engine;
    std::atomic<bool> engineStop{false};
    bool engineBusy = false;
    unsigned engineGen = 0;

    std::mutex mx;  // guards the strings below, written from worker threads
    std::string engineLine, pvLine, tbNote, tbStatus = "Endgame tables: loading...";
} g;

int S(int v) { return MulDiv(v, g.dpi, 96); }

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), &w[0], n);
    return w;
}

std::string exeDir() {
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p = buf;
    return p.substr(0, p.find_last_of("\\/"));
}

// ---------------------------------------------------------------- layout

struct Layout {
    int sq, ox, oy, panelX, panelW, height;
};

Layout layout() {
    RECT rc;
    GetClientRect(g.hwnd, &rc);
    Layout L;
    int margin = S(30);
    L.panelW = S(340);
    int avail = std::min<int>(rc.bottom - 2 * margin, rc.right - L.panelW - 3 * margin);
    L.sq = std::max(S(24), avail / 8);
    L.ox = margin;
    L.oy = std::max<int>(margin, (rc.bottom - 8 * L.sq) / 2);
    L.panelX = L.ox + 8 * L.sq + margin;
    L.height = rc.bottom;
    return L;
}

RECT squareRect(const Layout& L, int s) {
    int col = g.flipped ? 7 - fileOf(s) : fileOf(s);
    int row = g.flipped ? rankOf(s) : 7 - rankOf(s);
    RECT r = {L.ox + col * L.sq, L.oy + row * L.sq, 0, 0};
    r.right = r.left + L.sq;
    r.bottom = r.top + L.sq;
    return r;
}

int squareAt(const Layout& L, int x, int y) {
    if (x < L.ox || y < L.oy || x >= L.ox + 8 * L.sq || y >= L.oy + 8 * L.sq) return -1;
    int col = (x - L.ox) / L.sq, row = (y - L.oy) / L.sq;
    int f = g.flipped ? 7 - col : col;
    int r = g.flipped ? row : 7 - row;
    return r * 8 + f;
}

// ---------------------------------------------------------------- game state

bool engineToMove() { return g.gameOver.empty() && g.board.stm != g.human; }

void updateGameOver() {
    MoveList ml;
    g.board.genLegal(ml);
    g.gameOver.clear();
    if (!ml.count) {
        if (g.board.inCheck())
            g.gameOver = std::string("Checkmate - ") + (g.board.stm == WHITE ? "Black" : "White") + " wins";
        else
            g.gameOver = "Draw by stalemate";
    } else if (g.board.halfmove >= 100) {
        g.gameOver = "Draw by the 50-move rule";
    } else if (g.board.repetitionCount() >= 3) {
        g.gameOver = "Draw by threefold repetition";
    } else if (g.board.insufficientMaterial()) {
        g.gameOver = "Draw by insufficient material";
    }
}

void stopEngine() {
    if (g.engine.joinable()) {
        g.engineStop = true;
        g.engine.join();
    }
    g.engineBusy = false;
    g.engineGen++;  // drop any result already posted
}

void startEngine() {
    if (!engineToMove() || g.engineBusy) return;
    if (g.engine.joinable()) g.engine.join();
    g.engineBusy = true;
    g.engineStop = false;
    unsigned gen = ++g.engineGen;
    Board copy = g.board;
    int ms = THINK_TIMES[g.thinkIndex];
    bool engineWhite = g.board.stm == WHITE;
    HWND hwnd = g.hwnd;
    {
        std::lock_guard<std::mutex> lk(g.mx);
        g.engineLine = "Thinking...";
        g.pvLine.clear();
        g.tbNote.clear();
    }
    g.engine = std::thread([=] {
        Move m = findBestMove(copy, ms, &g.engineStop, [&](const SearchInfo& si) {
            int whiteScore = engineWhite ? si.score : -si.score;
            char buf[160];
            if (si.note.empty()) {
                double knps = si.ms > 0 ? double(si.nodes) / si.ms : 0;
                snprintf(buf, sizeof(buf), "Depth %d   Eval %s   %.2fM nodes   %.0f knps", si.depth,
                         scoreToString(whiteScore).c_str(), si.nodes / 1e6, knps);
            } else {
                snprintf(buf, sizeof(buf), "Eval %s", scoreToString(whiteScore).c_str());
            }
            std::lock_guard<std::mutex> lk(g.mx);
            g.engineLine = buf;
            g.pvLine = si.pv;
            g.tbNote = si.note;
            PostMessage(hwnd, WM_REFRESH, 0, 0);
        });
        PostMessage(hwnd, WM_ENGINE_DONE, gen, LPARAM(m));
    });
}

void applyMove(Move m) {
    g.san.push_back(g.board.moveToSAN(m));
    g.board.makeMove(m);
    g.played.push_back(m);
    g.selected = -1;
    g.targets.clear();
    updateGameOver();
    InvalidateRect(g.hwnd, nullptr, FALSE);
}

void resetHistory() {
    g.played.clear();
    g.san.clear();
    g.startStm = g.board.stm;
    g.startFullmove = g.board.fullmove;
    g.selected = -1;
    g.targets.clear();
    std::lock_guard<std::mutex> lk(g.mx);
    g.engineLine.clear();
    g.pvLine.clear();
    g.tbNote.clear();
}

void newGame(Color human) {
    stopEngine();
    clearSearchMemory();
    g.board.setStartPos();
    g.human = human;
    g.flipped = human == BLACK;
    resetHistory();
    updateGameOver();
    InvalidateRect(g.hwnd, nullptr, FALSE);
    startEngine();
}

void undo() {
    stopEngine();
    if (g.played.empty()) return;
    auto popOne = [] {
        g.board.unmakeMove();
        g.played.pop_back();
        g.san.pop_back();
    };
    popOne();
    if (!g.played.empty() && g.board.stm != g.human) popOne();
    g.selected = -1;
    g.targets.clear();
    updateGameOver();
    InvalidateRect(g.hwnd, nullptr, FALSE);
    startEngine();
}

void copyFEN() {
    std::string fen = g.board.getFEN();
    if (!OpenClipboard(g.hwnd)) return;
    EmptyClipboard();
    HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, fen.size() + 1);
    if (h) {
        memcpy(GlobalLock(h), fen.c_str(), fen.size() + 1);
        GlobalUnlock(h);
        SetClipboardData(CF_TEXT, h);
    }
    CloseClipboard();
}

void pasteFEN() {
    std::string text;
    if (!OpenClipboard(g.hwnd)) return;
    if (HANDLE h = GetClipboardData(CF_TEXT)) {
        if (const char* p = static_cast<const char*>(GlobalLock(h))) {
            text = p;
            GlobalUnlock(h);
        }
    }
    CloseClipboard();
    Board b;
    if (!b.setFEN(text)) {
        MessageBoxW(g.hwnd, L"The clipboard does not contain a valid FEN.", L"Paste FEN", MB_ICONWARNING);
        return;
    }
    stopEngine();
    g.board = b;
    g.human = b.stm;  // you play the side to move; switch sides to hand it to the engine
    g.flipped = g.human == BLACK;
    resetHistory();
    updateGameOver();
    InvalidateRect(g.hwnd, nullptr, FALSE);
}

void selectSquare(int s) {
    g.selected = -1;
    g.targets.clear();
    int p = s >= 0 ? g.board.sq[s] : EMPTY;
    if (!p || colorOf(p) != g.board.stm) return;
    MoveList ml;
    g.board.genLegal(ml);
    for (int i = 0; i < ml.count; i++)
        if (moveFrom(ml.moves[i]) == s) g.targets.push_back(ml.moves[i]);
    if (!g.targets.empty()) g.selected = s;
}

Move choosePromotion(const std::vector<Move>& options) {
    HMENU menu = CreatePopupMenu();
    const wchar_t* names[] = {L"", L"", L"Knight", L"Bishop", L"Rook", L"Queen"};
    for (int t : {QUEEN, ROOK, BISHOP, KNIGHT}) AppendMenuW(menu, MF_STRING, t, names[t]);
    POINT pt;
    GetCursorPos(&pt);
    int choice = TrackPopupMenu(menu, TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, g.hwnd, nullptr);
    DestroyMenu(menu);
    for (Move m : options)
        if (movePromo(m) == choice) return m;
    return NO_MOVE;
}

void onClick(int x, int y) {
    if (!g.gameOver.empty() || g.engineBusy || g.board.stm != g.human) return;
    Layout L = layout();
    int s = squareAt(L, x, y);
    if (s < 0) return;
    if (g.selected >= 0) {
        std::vector<Move> options;
        for (Move m : g.targets)
            if (moveTo(m) == s) options.push_back(m);
        if (!options.empty()) {
            Move m = options.size() > 1 ? choosePromotion(options) : options[0];
            if (m) {
                applyMove(m);
                startEngine();
            }
            return;
        }
    }
    selectSquare(s == g.selected ? -1 : s);
    InvalidateRect(g.hwnd, nullptr, FALSE);
}

// ---------------------------------------------------------------- painting

void fillRect(HDC dc, const RECT& r, COLORREF c) {
    HBRUSH b = CreateSolidBrush(c);
    FillRect(dc, &r, b);
    DeleteObject(b);
}

void ellipse(HDC dc, int cx, int cy, int rad, COLORREF fill, COLORREF pen, int penW) {
    HBRUSH b = fill == CLR_INVALID ? (HBRUSH)GetStockObject(NULL_BRUSH) : CreateSolidBrush(fill);
    HPEN p = penW ? CreatePen(PS_SOLID, penW, pen) : (HPEN)GetStockObject(NULL_PEN);
    HGDIOBJ ob = SelectObject(dc, b), op = SelectObject(dc, p);
    Ellipse(dc, cx - rad, cy - rad, cx + rad + 1, cy + rad + 1);
    SelectObject(dc, ob);
    SelectObject(dc, op);
    if (fill != CLR_INVALID) DeleteObject(b);
    if (penW) DeleteObject(p);
}

HFONT makeFont(const wchar_t* face, int px, int weight = FW_NORMAL) {
    return CreateFontW(-px, 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                       CLEARTYPE_QUALITY, DEFAULT_PITCH, face);
}

void drawText(HDC dc, const std::wstring& s, RECT r, COLORREF c, UINT fmt) {
    SetTextColor(dc, c);
    DrawTextW(dc, s.c_str(), int(s.size()), &r, fmt | DT_NOPREFIX);
}

void paintBoard(HDC dc, const Layout& L) {
    Move last = g.played.empty() ? NO_MOVE : g.played.back();
    int checkSq = g.board.inCheck() ? g.board.kingSq[g.board.stm] : -1;
    for (int s = 0; s < 64; s++) {
        RECT r = squareRect(L, s);
        bool light = (fileOf(s) + rankOf(s)) & 1;
        COLORREF c = light ? C_LIGHT : C_DARK;
        if (last && (s == moveFrom(last) || s == moveTo(last))) c = light ? C_LAST_LIGHT : C_LAST_DARK;
        if (s == g.selected) c = C_SELECT;
        fillRect(dc, r, c);
        if (s == checkSq) ellipse(dc, (r.left + r.right) / 2, (r.top + r.bottom) / 2, L.sq * 9 / 20, C_CHECK, 0, 0);
    }

    // Coordinates.
    HFONT coord = makeFont(L"Segoe UI", std::max(S(11), L.sq / 6), FW_SEMIBOLD);
    HGDIOBJ old = SelectObject(dc, coord);
    for (int i = 0; i < 8; i++) {
        int f = g.flipped ? 7 - i : i;
        RECT fr = {L.ox + i * L.sq, L.oy + 8 * L.sq + S(4), L.ox + (i + 1) * L.sq, L.oy + 8 * L.sq + S(26)};
        drawText(dc, std::wstring(1, wchar_t('a' + f)), fr, C_DIM_TEXT, DT_CENTER | DT_TOP | DT_SINGLELINE);
        int r = g.flipped ? i : 7 - i;
        RECT rr = {L.ox - S(26), L.oy + i * L.sq, L.ox - S(6), L.oy + (i + 1) * L.sq};
        drawText(dc, std::wstring(1, wchar_t('1' + r)), rr, C_DIM_TEXT, DT_RIGHT | DT_VCENTER | DT_SINGLELINE);
    }
    SelectObject(dc, old);
    DeleteObject(coord);

    // Pieces: a solid glyph for the body and, for white, the outline glyph on top for detail.
    const wchar_t SOLID[7] = {0, 0x265F, 0x265E, 0x265D, 0x265C, 0x265B, 0x265A};
    const wchar_t OUTLINE[7] = {0, 0x2659, 0x2658, 0x2657, 0x2656, 0x2655, 0x2654};
    HFONT pf = makeFont(L"Segoe UI Symbol", L.sq * 4 / 5);
    old = SelectObject(dc, pf);
    for (int s = 0; s < 64; s++) {
        int p = g.board.sq[s];
        if (!p) continue;
        RECT r = squareRect(L, s);
        OffsetRect(&r, 0, L.sq / 40);
        int t = typeOf(p);
        UINT fmt = DT_CENTER | DT_VCENTER | DT_SINGLELINE;
        if (colorOf(p) == WHITE) {
            drawText(dc, std::wstring(1, SOLID[t]), r, RGB(255, 255, 255), fmt);
            drawText(dc, std::wstring(1, OUTLINE[t]), r, RGB(25, 25, 25), fmt);
        } else {
            drawText(dc, std::wstring(1, SOLID[t]), r, RGB(25, 25, 25), fmt);
        }
    }
    SelectObject(dc, old);
    DeleteObject(pf);

    // Legal destination hints.
    for (Move m : g.targets) {
        RECT r = squareRect(L, moveTo(m));
        int cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2;
        if (g.board.sq[moveTo(m)] || (moveFlags(m) & MF_EP))
            ellipse(dc, cx, cy, L.sq * 9 / 20, CLR_INVALID, C_HINT, std::max(2, L.sq / 14));
        else
            ellipse(dc, cx, cy, L.sq / 7, C_HINT, 0, 0);
    }
}

void paintPanel(HDC dc, const Layout& L) {
    std::string engineLine, pvLine, tbNote, tbStatus;
    {
        std::lock_guard<std::mutex> lk(g.mx);
        engineLine = g.engineLine;
        pvLine = g.pvLine;
        tbNote = g.tbNote;
        tbStatus = g.tbStatus;
    }
    int x = L.panelX, w = L.panelW, y = L.oy;
    HFONT title = makeFont(L"Segoe UI", S(26), FW_BOLD);
    HFONT bold = makeFont(L"Segoe UI", S(16), FW_SEMIBOLD);
    HFONT normal = makeFont(L"Segoe UI", S(14));
    HFONT mono = makeFont(L"Consolas", S(15));
    auto line = [&](HFONT f, const std::string& s, COLORREF c, int h, bool wrap = false) {
        SelectObject(dc, f);
        RECT r = {x, y, x + w, y + h};
        if (wrap) {
            RECT calc = r;
            DrawTextW(dc, widen(s).c_str(), -1, &calc, DT_WORDBREAK | DT_CALCRECT | DT_NOPREFIX);
            r.bottom = std::max(r.bottom, calc.bottom);
        }
        drawText(dc, widen(s), r, c, wrap ? DT_WORDBREAK : (DT_SINGLELINE | DT_END_ELLIPSIS));
        y = r.bottom;
    };
    HGDIOBJ old = SelectObject(dc, title);

    line(title, "ChessBot", C_PANEL_TEXT, S(36));
    std::string sides = std::string("You: ") + (g.human == WHITE ? "White" : "Black") +
                        "    Engine: " + (g.human == WHITE ? "Black" : "White") + "    (" +
                        std::to_string(THINK_TIMES[g.thinkIndex] / 1000) + "s/move)";
    line(normal, sides, C_DIM_TEXT, S(24));
    y += S(10);

    std::string status;
    COLORREF statusColor = C_ACCENT;
    if (!g.gameOver.empty())
        status = g.gameOver, statusColor = RGB(240, 190, 90);
    else if (g.engineBusy)
        status = "Engine is thinking...";
    else if (g.board.stm == g.human)
        status = std::string("Your move (") + (g.human == WHITE ? "White" : "Black") + ")";
    else
        status = "Engine to move";
    line(bold, status, statusColor, S(26));
    y += S(8);

    if (!engineLine.empty()) line(normal, engineLine, C_PANEL_TEXT, S(22));
    if (!tbNote.empty()) line(normal, tbNote, C_ACCENT, S(22));
    if (!pvLine.empty()) line(normal, "PV: " + pvLine, C_DIM_TEXT, S(20), true);
    y += S(12);

    // Move list, two plies per row, showing the most recent rows that fit.
    line(bold, "Moves", C_PANEL_TEXT, S(26));
    std::vector<std::string> rows;
    int num = g.startFullmove;
    size_t i = 0;
    if (g.startStm == BLACK && !g.san.empty()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%3d. %-9s%s", num++, "...", g.san[0].c_str());
        rows.push_back(buf);
        i = 1;
    }
    for (; i < g.san.size(); i += 2) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%3d. %-9s%s", num++, g.san[i].c_str(),
                 i + 1 < g.san.size() ? g.san[i + 1].c_str() : "");
        rows.push_back(buf);
    }
    int footer = S(48);
    int rowH = S(20);
    int fit = std::max(1, (L.height - footer - y) / rowH);
    size_t first = rows.size() > size_t(fit) ? rows.size() - fit : 0;
    for (size_t k = first; k < rows.size(); k++) line(mono, rows[k], C_PANEL_TEXT, rowH);

    y = L.height - footer + S(8);
    line(normal, tbStatus, C_DIM_TEXT, S(20));

    SelectObject(dc, old);
    for (HFONT f : {title, bold, normal, mono}) DeleteObject(f);
}

void paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    HDC dc = CreateCompatibleDC(hdc);
    HBITMAP bmp = CreateCompatibleBitmap(hdc, rc.right, rc.bottom);
    HGDIOBJ oldBmp = SelectObject(dc, bmp);
    SetBkMode(dc, TRANSPARENT);
    fillRect(dc, rc, C_BG);
    Layout L = layout();
    paintBoard(dc, L);
    paintPanel(dc, L);
    BitBlt(hdc, 0, 0, rc.right, rc.bottom, dc, 0, 0, SRCCOPY);
    SelectObject(dc, oldBmp);
    DeleteObject(bmp);
    DeleteDC(dc);
    EndPaint(hwnd, &ps);
}

// ---------------------------------------------------------------- window

void updateTimeMenu() {
    HMENU menu = GetMenu(g.hwnd);
    int n = int(sizeof(THINK_TIMES) / sizeof(THINK_TIMES[0]));
    CheckMenuRadioItem(menu, ID_TIME_BASE, ID_TIME_BASE + n - 1, ID_TIME_BASE + g.thinkIndex, MF_BYCOMMAND);
}

HMENU buildMenu() {
    HMENU bar = CreateMenu();
    HMENU game = CreatePopupMenu();
    AppendMenuW(game, MF_STRING, ID_NEW_WHITE, L"New Game as &White\tCtrl+N");
    AppendMenuW(game, MF_STRING, ID_NEW_BLACK, L"New Game as &Black\tCtrl+B");
    AppendMenuW(game, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(game, MF_STRING, ID_UNDO, L"&Undo Move\tCtrl+Z");
    AppendMenuW(game, MF_STRING, ID_FLIP, L"&Flip Board\tF");
    AppendMenuW(game, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(game, MF_STRING, ID_COPY_FEN, L"&Copy FEN\tCtrl+C");
    AppendMenuW(game, MF_STRING, ID_PASTE_FEN, L"&Paste FEN\tCtrl+V");
    AppendMenuW(game, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(game, MF_STRING, ID_EXIT, L"E&xit");
    AppendMenuW(bar, MF_POPUP, UINT_PTR(game), L"&Game");

    HMENU eng = CreatePopupMenu();
    AppendMenuW(eng, MF_STRING, ID_SWITCH_SIDES, L"&Switch Sides\tCtrl+E");
    AppendMenuW(eng, MF_STRING, ID_MOVE_NOW, L"&Move Now\tSpace");
    AppendMenuW(eng, MF_SEPARATOR, 0, nullptr);
    const wchar_t* labels[] = {L"Think 1 second", L"Think 3 seconds", L"Think 5 seconds", L"Think 10 seconds",
                               L"Think 30 seconds"};
    for (int i = 0; i < 5; i++) AppendMenuW(eng, MF_STRING, ID_TIME_BASE + i, labels[i]);
    AppendMenuW(bar, MF_POPUP, UINT_PTR(eng), L"&Engine");
    return bar;
}

void onCommand(int id) {
    switch (id) {
        case ID_NEW_WHITE: newGame(WHITE); break;
        case ID_NEW_BLACK: newGame(BLACK); break;
        case ID_UNDO: undo(); break;
        case ID_FLIP: g.flipped = !g.flipped; InvalidateRect(g.hwnd, nullptr, FALSE); break;
        case ID_COPY_FEN: copyFEN(); break;
        case ID_PASTE_FEN: pasteFEN(); break;
        case ID_EXIT: DestroyWindow(g.hwnd); break;
        case ID_SWITCH_SIDES:
            stopEngine();
            g.human = Color(g.human ^ 1);
            g.flipped = g.human == BLACK;
            g.selected = -1;
            g.targets.clear();
            InvalidateRect(g.hwnd, nullptr, FALSE);
            startEngine();
            break;
        case ID_MOVE_NOW: g.engineStop = true; break;
        default:
            if (id >= ID_TIME_BASE && id < ID_TIME_BASE + 5) {
                g.thinkIndex = id - ID_TIME_BASE;
                updateTimeMenu();
                InvalidateRect(g.hwnd, nullptr, FALSE);
            }
    }
}

LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_PAINT: paint(hwnd); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_SIZE: InvalidateRect(hwnd, nullptr, FALSE); return 0;
        case WM_GETMINMAXINFO: {
            auto* mm = reinterpret_cast<MINMAXINFO*>(lp);
            mm->ptMinTrackSize = {S(820), S(520)};
            return 0;
        }
        case WM_LBUTTONDOWN: onClick(GET_X_LPARAM(lp), GET_Y_LPARAM(lp)); return 0;
        case WM_RBUTTONDOWN:
            g.selected = -1;
            g.targets.clear();
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        case WM_COMMAND: onCommand(LOWORD(wp)); return 0;
        case WM_KEYDOWN: {
            bool ctrl = GetKeyState(VK_CONTROL) < 0;
            if (ctrl && wp == 'N') onCommand(ID_NEW_WHITE);
            else if (ctrl && wp == 'B') onCommand(ID_NEW_BLACK);
            else if (ctrl && wp == 'Z') onCommand(ID_UNDO);
            else if (ctrl && wp == 'C') onCommand(ID_COPY_FEN);
            else if (ctrl && wp == 'V') onCommand(ID_PASTE_FEN);
            else if (ctrl && wp == 'E') onCommand(ID_SWITCH_SIDES);
            else if (!ctrl && wp == 'F') onCommand(ID_FLIP);
            else if (wp == VK_SPACE) onCommand(ID_MOVE_NOW);
            return 0;
        }
        case WM_REFRESH: InvalidateRect(hwnd, nullptr, FALSE); return 0;
        case WM_ENGINE_DONE: {
            if (unsigned(wp) != g.engineGen) return 0;  // stale result
            if (g.engine.joinable()) g.engine.join();
            g.engineBusy = false;
            Move m = Move(lp);
            MoveList ml;
            g.board.genLegal(ml);
            for (int i = 0; i < ml.count; i++)
                if (ml.moves[i] == m) {
                    applyMove(m);
                    break;
                }
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        case WM_DESTROY:
            stopEngine();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int show) {
    SetProcessDPIAware();
    initTables();
    {
        HDC screen = GetDC(nullptr);
        g.dpi = GetDeviceCaps(screen, LOGPIXELSY);
        ReleaseDC(nullptr, screen);
    }

    WNDCLASSW wc = {};
    wc.lpfnWndProc = wndProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    wc.lpszClassName = L"ChessBotWindow";
    RegisterClassW(&wc);

    g.hwnd = CreateWindowW(wc.lpszClassName, L"ChessBot", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                           S(1120), S(800), nullptr, buildMenu(), inst, nullptr);
    updateTimeMenu();

    // Load (or build, on first run) the endgame tables in the background.
    tb::init(exeDir() + "\\tb");
    HWND hwnd = g.hwnd;
    std::thread([hwnd] {
        tb::loadOrGenerateAll([hwnd](const std::string& s) {
            {
                std::lock_guard<std::mutex> lk(g.mx);
                g.tbStatus = "Endgame tables: " + s;
            }
            PostMessage(hwnd, WM_REFRESH, 0, 0);
        });
    }).detach();

    newGame(WHITE);
    ShowWindow(g.hwnd, show);
    UpdateWindow(g.hwnd);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    return 0;
}
