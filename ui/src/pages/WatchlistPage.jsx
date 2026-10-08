/**
 * Watchlist — locally persisted (UI-level) symbol tracking.
 *
 * This is explicitly TEMPORARY FRONTEND STATE (localStorage), kept
 * clearly separate from any future account-backed watchlist.
 * Live prices per symbol are intentionally NOT fabricated here;
 * each row links to the real analysis page for real data.
 */

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import { searchStocks } from "../database/StockDatabase";

const STORAGE_KEY = "stockvibe.watchlist.v1";
const MAX_SYMBOLS = 30;

const WatchlistContext = createContext(null);

const readStored = () => {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((s) => typeof s === "string") : [];
  } catch {
    return [];
  }
};

export function WatchlistProvider({ children }) {
  const [symbols, setSymbols] = useState(readStored);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(symbols));
    } catch {
      /* storage unavailable (private mode) — watchlist stays in-memory */
    }
  }, [symbols]);

  const addSymbol = useCallback(
    (raw) => {
      const symbol = String(raw || "").trim().toUpperCase();
      if (!symbol) return { ok: false, reason: "empty" };
      if (symbols.includes(symbol)) return { ok: false, reason: "duplicate" };
      if (symbols.length >= MAX_SYMBOLS) return { ok: false, reason: "limit" };
      setSymbols((prev) => [...prev, symbol]);
      return { ok: true };
    },
    [symbols]
  );

  const removeSymbol = useCallback((symbol) => {
    setSymbols((prev) => prev.filter((s) => s !== symbol));
  }, []);

  return (
    <WatchlistContext.Provider value={{ symbols, addSymbol, removeSymbol }}>
      {children}
    </WatchlistContext.Provider>
  );
}

export function useWatchlist() {
  const ctx = useContext(WatchlistContext);
  if (!ctx) {
    throw new Error("useWatchlist must be used inside <WatchlistProvider>");
  }
  return ctx;
}

/* ─── Page ─── */

const WatchlistPage = () => {
  const { symbols, addSymbol, removeSymbol } = useWatchlist();
  const [input, setInput] = useState("");
  const [feedback, setFeedback] = useState(null);

  const suggestions =
    input.trim().length >= 1 ? searchStocks(input.trim(), 6) : [];

  const submit = (e) => {
    e.preventDefault();
    if (suggestions.length > 0 && !suggestions.some((s) => s.symbol === input.trim().toUpperCase())) {
      // Use top suggestion when user typed a partial name
      const result = addSymbol(suggestions[0].symbol);
      announce(result, suggestions[0].symbol);
      setInput("");
    } else {
      const result = addSymbol(input);
      announce(result, input.trim().toUpperCase());
      if (result.ok) setInput("");
    }
  };

  const announce = (result, symbol) => {
    if (result.ok) setFeedback(`Added ${symbol}.`);
    else if (result.reason === "duplicate") setFeedback(`${symbol} is already in your watchlist.`);
    else if (result.reason === "limit") setFeedback(`Watchlist is full (${MAX_SYMBOLS} symbols).`);
    else setFeedback("Enter a symbol to add.");
  };

  return (
    <div className="sv-page">
      <div className="sv-page-header">
        <h1 className="sv-page-title">Watchlist</h1>
        <p className="sv-page-subtitle">
          Symbols you're tracking. Stored locally in this browser — an
          account-backed watchlist arrives with user profiles.
        </p>
      </div>

      <section className="sv-card">
        <form onSubmit={submit} className="sv-watch-add-form">
          <label htmlFor="wl-input" id="wl-add" className="sv-metric-label">
            Add symbol
          </label>
          <div className="sv-watch-add-row">
            <input
              id="wl-input"
              type="text"
              className="sv-search-input"
              placeholder="e.g. AAPL or Apple"
              value={input}
              onChange={(e) => {
                setInput(e.target.value.toUpperCase());
                setFeedback(null);
              }}
              maxLength={12}
              autoComplete="off"
              list="wl-suggestions"
            />
            <datalist id="wl-suggestions">
              {suggestions.map((s) => (
                <option key={s.symbol} value={s.symbol}>
                  {s.name}
                </option>
              ))}
            </datalist>
            <button type="submit" className="sv-btn sv-btn-primary">
              Add
            </button>
          </div>
          {feedback && (
            <p className="sv-card-subtitle" role="status" aria-live="polite">
              {feedback}
            </p>
          )}
        </form>
      </section>

      <section className="sv-card" style={{ marginTop: "var(--sv-space-6)" }} aria-labelledby="wl-list">
        <div className="sv-card-header">
          <h2 id="wl-list" className="sv-card-title">
            Your symbols
          </h2>
          <span className="sv-badge">{symbols.length} tracked</span>
        </div>

        {symbols.length === 0 ? (
          <div className="sv-empty">
            <p className="sv-empty-title">Nothing here yet.</p>
            <p className="sv-empty-text">
              Add a symbol above to start tracking it.
            </p>
          </div>
        ) : (
          <ul className="sv-watch-row-list">
            {symbols.map((symbol) => (
              <li key={symbol}>
                <div className="sv-watch-row">
                  <Link
                    to={`/stocks/${encodeURIComponent(symbol)}`}
                    className="sv-watch-row-link"
                  >
                    <span className="sv-num sv-watch-symbol">{symbol}</span>
                    <span className="sv-watch-meta">Open analysis</span>
                  </Link>
                  <button
                    type="button"
                    className="sv-btn sv-btn-ghost sv-btn-sm"
                    onClick={() => removeSymbol(symbol)}
                    aria-label={`Remove ${symbol} from watchlist`}
                  >
                    Remove
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};

export default WatchlistPage;
