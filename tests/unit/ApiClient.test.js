/**
 * Centralized API client behavior tests.
 */

import { toUserMessage, errorStatus } from "../../ui/src/api/client";

describe("API client error normalization", () => {
  test("network errors produce a safe connection message", () => {
    expect(toUserMessage({ code: "ECONNABORTED", name: "AbortError" })).toContain(
      "timed out"
    );
    expect(toUserMessage({ code: "ERR_NETWORK" })).toContain("Cannot reach");
  });

  test("HTTP status errors map to human messages", () => {
    expect(
      toUserMessage({ response: { status: 404, data: {} } })
    ).toContain("couldn't find");
    expect(
      toUserMessage({ response: { status: 500, data: {} } })
    ).toContain("server");
    expect(
      toUserMessage({ response: { status: 429, data: {} } })
    ).toContain("Too many requests");
  });

  test("server-provided short error messages pass through", () => {
    expect(
      toUserMessage({ response: { status: 400, data: { error: "Invalid ticker format." } } })
    ).toBe("Invalid ticker format.");
  });

  test("long/technical error text is never surfaced raw", () => {
    const noisy =
      "Traceback (most recent call last): FileNotFoundError: '/models/x.pkl' " +
      "raised at django.core.handlers.exception";
    const result = toUserMessage({ response: { status: 500, data: { error: noisy } } });
    expect(result).not.toContain("Traceback");
    expect(result).not.toContain("django");
  });

  test("errorStatus extracts typed status", () => {
    expect(errorStatus({ response: { status: 404 } })).toBe(404);
    expect(errorStatus({ code: "ECONNABORTED" })).toBe("timeout");
    expect(errorStatus({ code: "ERR_NETWORK" })).toBe("network");
  });
});
