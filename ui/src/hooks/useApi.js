/**
 * useApi — small declarative data-fetching hook for the API layer.
 *
 * Usage:
 *   const { data, loading, error, refetch } = useApi(
 *     () => getMarketOverview(),
 *     { loadOnMount: true }
 *   );
 *
 * `fetcher` receives a `signal`-free request from the centralized
 * client; cancellation of in-flight requests on unmount is handled
 * by ignoring stale responses (isStale pattern below).
 */

import { useState, useEffect, useCallback, useRef } from "react";

const IDLE = "idle";
const LOADING = "loading";
const SUCCESS = "success";
const ERROR = "error";

export default function useApi(fetcher, { loadOnMount = true, deps = [] } = {}) {
  const [state, setState] = useState({ status: IDLE, data: null, error: null });
  const mountedRef = useRef(true);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const run = useCallback(async () => {
    setState((prev) => ({ ...prev, status: LOADING, error: null }));
    const result = await fetcherRef.current();
    if (!mountedRef.current) return;
    if (result.ok) {
      setState({ status: SUCCESS, data: result.data, error: null });
    } else {
      setState((prev) => ({ ...prev, status: ERROR, error: result.error }));
    }
  }, []);

  // Re-run when `deps` change (e.g. symbol / timeframe). fetcherRef always
  // holds the latest closure, so run() picks up new arguments automatically.
  useEffect(() => {
    if (loadOnMount) run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return {
    data: state.data,
    loading: state.status === LOADING || state.status === IDLE,
    error: state.status === ERROR ? state.error : null,
    refetch: run,
  };
}
