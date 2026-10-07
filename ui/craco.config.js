const path = require("path");

module.exports = {
  babel: {
    presets: [
      [
        "@babel/preset-env",
        {
          targets: { node: "current" },
          loose: true,
        },
      ],
      [
        "@babel/preset-react",
        {
          runtime: "automatic",
        },
      ],
    ],
    plugins: [
      ["@babel/plugin-transform-private-methods", { loose: true }],
      ["@babel/plugin-transform-private-property-in-object", { loose: true }],
      ["@babel/plugin-transform-class-properties", { loose: true }],
    ],
  },

  jest: {
    configure: (jestConfig) => {
      jestConfig.roots = ["<rootDir>/../tests"];
      jestConfig.testRegex = ".*\\.(test|spec)\\.(js|jsx)$";
      delete jestConfig.testMatch;

      jestConfig.setupFilesAfterEnv = ["<rootDir>/../tests/SetupTests.jsx"];
      jestConfig.testEnvironment = "jsdom";

      // Enhanced module resolution
      jestConfig.moduleDirectories = [
        "node_modules",
        "<rootDir>/node_modules",
        "<rootDir>/../node_modules", // Look in parent directory
        path.resolve(__dirname, "node_modules"), // Absolute path to ui/node_modules
      ];

      jestConfig.moduleNameMapper = {
        ...jestConfig.moduleNameMapper,
        "\\.(css|less|scss|sass)$": "identity-obj-proxy",
        "\\.(gif|ttf|eot|svg|png|jpg|jpeg)$": "jest-transform-stub",
        // Add explicit mappings for testing library
        "^@testing-library/jest-dom$":
          "<rootDir>/node_modules/@testing-library/jest-dom",
        "^@testing-library/react$":
          "<rootDir>/node_modules/@testing-library/react",
        "^@testing-library/user-event$":
          "<rootDir>/node_modules/@testing-library/user-event",
      };

      jestConfig.transform = {
        "^.+\\.(js|jsx)$": [
          "babel-jest",
          {
            presets: [
              [
                "@babel/preset-env",
                {
                  targets: { node: "current" },
                  loose: true,
                },
              ],
              [
                "@babel/preset-react",
                {
                  runtime: "automatic",
                },
              ],
            ],
            plugins: [
              ["@babel/plugin-transform-private-methods", { loose: true }],
              [
                "@babel/plugin-transform-private-property-in-object",
                { loose: true },
              ],
              ["@babel/plugin-transform-class-properties", { loose: true }],
            ],
          },
        ],
      };

      jestConfig.moduleFileExtensions = ["js", "jsx", "json", "node"];
      jestConfig.testPathIgnorePatterns = [
        "/node_modules/",
        "<rootDir>/../tests/backend/",
      ];
      jestConfig.clearMocks = true;

      return jestConfig;
    },
  },

  // ... rest of your webpack config ...
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.entry = path.resolve(__dirname, "src/index.jsx");

      const htmlWebpackPlugin = webpackConfig.plugins.find(
        (plugin) => plugin.constructor.name === "HtmlWebpackPlugin"
      );

      if (htmlWebpackPlugin) {
        htmlWebpackPlugin.options.template = path.resolve(
          __dirname,
          "public/index.html"
        );
      }

      webpackConfig.resolve.modules = [
        path.resolve(__dirname, "src"),
        path.resolve(__dirname, "src/components"),
        "node_modules",
      ];

      webpackConfig.resolve.alias = {
        ...webpackConfig.resolve.alias,
        "@": path.resolve(__dirname, "src"),
        "@components": path.resolve(__dirname, "src/components"),
      };

      const babelLoaderRule = webpackConfig.module.rules.find(
        (rule) =>
          rule.oneOf &&
          rule.oneOf.some(
            (oneOf) => oneOf.test && oneOf.test.toString().includes("js")
          )
      );

      if (babelLoaderRule) {
        const jsRule = babelLoaderRule.oneOf.find(
          (oneOf) => oneOf.test && oneOf.test.toString().includes("js")
        );

        if (jsRule) {
          jsRule.include = [
            path.resolve(__dirname, "src"),
            path.resolve(__dirname, "src/components"),
          ];
        }
      }

      return webpackConfig;
    },
  },

  devServer: {
    static: {
      directory: path.join(__dirname, "public"),
    },
    port: 3000,
    open: true,
  },
};
