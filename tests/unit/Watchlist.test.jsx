/**
 * Watchlist behavior tests — add, duplicate rejection, persistence.
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { MemoryRouter } from "react-router-dom";
import WatchlistPage, {
  WatchlistProvider,
  useWatchlist,
} from "../../ui/src/pages/WatchlistPage";

// localStorage is globally mocked in SetupTests; give it a real backing store
beforeAll(() => {
  const store = new Map();
  global.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    clear: () => store.clear(),
  };
});

const renderWatchlist = () =>
  render(
    <MemoryRouter>
      <WatchlistProvider>
        <WatchlistPage />
      </WatchlistProvider>
    </MemoryRouter>
  );

describe("Watchlist", () => {
  test("starts empty with an honest empty state", () => {
    renderWatchlist();
    expect(screen.getByText("Nothing here yet.")).toBeInTheDocument();
  });

  test("adds a symbol and shows it in the list", () => {
    renderWatchlist();

    const input = screen.getByLabelText("Add symbol");
    fireEvent.change(input, { target: { value: "NVDA" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    expect(screen.getByText("Added NVDA.")).toBeInTheDocument();
    expect(screen.getAllByText("NVDA").length).toBeGreaterThan(0);
  });

  test("rejects duplicate symbols with feedback", () => {
    renderWatchlist();

    const input = screen.getByLabelText("Add symbol");
    fireEvent.change(input, { target: { value: "AAPL" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    fireEvent.change(input, { target: { value: "AAPL" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    expect(
      screen.getByText("AAPL is already in your watchlist.")
    ).toBeInTheDocument();
  });

  test("removes a symbol via the remove button", () => {
    renderWatchlist();

    const input = screen.getByLabelText("Add symbol");
    fireEvent.change(input, { target: { value: "TSLA" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));

    fireEvent.click(screen.getByRole("button", { name: "Remove TSLA from watchlist" }));

    expect(screen.getByText("Nothing here yet.")).toBeInTheDocument();
  });
});
