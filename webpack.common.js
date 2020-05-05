const path = require('path');

module.exports = {
  entry: {
    index: './src/index.ts',
  },
  externals: {
    opencv: 'cv',
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
      },
      {
        test: /\.css$/,
        use: [
          "style-loader",
          {
            loader: "css-loader",
            options: { url: false, },
          }
        ],
      },
      {
        test: /\.(png|jpe?g|gif|svg)$/,
        use: "file-loader?name=/img/[name].[ext]",
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  output: {
    path: path.join(__dirname, 'public'),
    publicPath: '/js/',
    filename: '[name].js',
    libraryTarget: 'umd',
  },
};
