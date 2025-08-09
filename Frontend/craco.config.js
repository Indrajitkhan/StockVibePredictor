const path = require("path");

module.exports = {
  babel: {
    presets: [
      "@babel/preset-env",
      ["@babel/preset-react", { runtime: "automatic" }],
    ],
  },
  webpack: {
    configure: (webpackConfig) => {
      webpackConfig.entry = path.resolve(__dirname, "Src/index.js");

      const htmlWebpackPlugin = webpackConfig.plugins.find(
        (plugin) => plugin.constructor.name === "HtmlWebpackPlugin"
      );

      if (htmlWebpackPlugin) {
        htmlWebpackPlugin.options.template = path.resolve(
          __dirname,
          "Public/index.html"
        );
      }

      webpackConfig.resolve.modules = [
        path.resolve(__dirname, "Src"),
        path.resolve(__dirname, "Components"),
        "node_modules",
      ];

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
            path.resolve(__dirname, "Src"),
            path.resolve(__dirname, "Components"),
          ];
        }
      }

      return webpackConfig;
    },
  },
  devServer: {
    static: {
      directory: path.join(__dirname, "Public"),
    },
    port: 3000,
    open: true,
  },
};
