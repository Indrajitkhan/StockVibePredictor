import "@testing-library/jest-dom";

// Mock Chart.js
jest.mock(
  "chart.js",
  () => ({
    Chart: {
      register: jest.fn(),
    },
    ArcElement: jest.fn(),
    Tooltip: jest.fn(),
    Legend: jest.fn(),
    Title: jest.fn(),
    CategoryScale: jest.fn(),
    LinearScale: jest.fn(),
    BarElement: jest.fn(),
    LineElement: jest.fn(),
    PointElement: jest.fn(),
  }),
  { virtual: true }
);

// Mock react-chartjs-2
jest.mock(
  "react-chartjs-2",
  () => ({
    Doughnut: ({ data, options }) => (
      <div
        data-testid="doughnut-chart"
        data-chart-data={JSON.stringify(data)}
        data-options={JSON.stringify(options)}
      />
    ),
    Bar: ({ data, options }) => (
      <div
        data-testid="bar-chart"
        data-chart-data={JSON.stringify(data)}
        data-options={JSON.stringify(options)}
      />
    ),
    Line: ({ data, options }) => (
      <div
        data-testid="line-chart"
        data-chart-data={JSON.stringify(data)}
        data-options={JSON.stringify(options)}
      />
    ),
  }),
  { virtual: true }
);

// DON'T mock CSS here - let moduleNameMapper handle it
// jest.mock("*.css", () => ({})); // REMOVED THIS LINE

// Mock axios — the shared mock instance backs the centralized API client
// (client.js calls axios.create().request/get/post).
// NOTE: the variable must be named mock* — jest.mock factories may only
// reference out-of-scope variables whose names begin with "mock".
const mockAxios = {
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} })),
  request: jest.fn(() => Promise.resolve({ data: {} })),
};

jest.mock(
  "axios",
  () => ({
    __esModule: true,
    default: {
      ...mockAxios,
      create: jest.fn(() => ({
        ...mockAxios,
        defaults: { headers: { common: {} } },
        interceptors: {
          request: { use: jest.fn(), eject: jest.fn() },
          response: { use: jest.fn(), eject: jest.fn() },
        },
      })),
    },
    ...mockAxios,
  }),
  { virtual: true }
);

// Global mocks
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
  })
);

global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock sessionStorage
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.sessionStorage = sessionStorageMock;

// Clean up after each test
// Mark the test environment as an act() environment (React 18 requirement)
global.IS_REACT_ACT_ENVIRONMENT = true;

afterEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});
