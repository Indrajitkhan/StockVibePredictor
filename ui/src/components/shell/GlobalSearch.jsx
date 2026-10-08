/**
 * Global search — symbol/company lookup over the local StockDatabase.
 * Selecting a result navigates to /stocks/:symbol.
 */

import React, { useState, useRef, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { searchStocks } from "../../database/StockDatabase";

const GlobalSearch = () => {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(0);
  const navigate = useNavigate();
  const rootRef = useRef(null);

  const results = useMemo(
    () => (query.trim().length >= 1 ? searchStocks(query.trim(), 8) : []),
    [query]
  );

  useEffect(() => {
    const onDocClick = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const go = (symbol) => {
    navigate(`/stocks/${symbol}`);
    setQuery("");
    setOpen(false);
  };

  const onSubmit = (e) => {
    e.preventDefault();
    const exact = results.find(
      (r) => r.symbol.toLowerCase() === query.trim().toLowerCase()
    );
    if (exact) {
      go(exact.symbol);
    } else if (results.length > 0) {
      go(results[highlighted].symbol);
    } else if (query.trim()) {
      // Not in the local database — let the analysis page validate
      // against the API rather than blocking the user here.
      go(query.trim().toUpperCase());
    }
  };

  const onKeyDown = (e) => {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlighted((h) => (h + 1) % results.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlighted((h) => (h - 1 + results.length) % results.length);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="sv-search" ref={rootRef}>
      <form onSubmit={onSubmit} role="search">
        <span className="sv-search-icon" aria-hidden="true">
          ⌕
        </span>
        <input
          type="text"
          className="sv-search-input"
          placeholder="Search symbol or company"
          aria-label="Search stocks by symbol or company name"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
            setHighlighted(0);
          }}
          onFocus={() => results.length > 0 && setOpen(true)}
          onKeyDown={onKeyDown}
          maxLength={12}
          autoComplete="off"
        />
      </form>

      {open && results.length > 0 && (
        <ul className="sv-search-results" role="listbox" aria-label="Search suggestions">
          {results.map((r, i) => (
            <li key={`${r.symbol}-${i}`} role="option" aria-selected={i === highlighted}>
              <button
                type="button"
                className={`sv-search-item${i === highlighted ? " highlighted" : ""}`}
                onMouseEnter={() => setHighlighted(i)}
                onClick={() => go(r.symbol)}
              >
                <span className="sv-num sv-search-symbol">{r.symbol}</span>
                <span className="sv-search-name">{r.name}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default GlobalSearch;
