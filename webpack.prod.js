const path = require('path');
const merge = require('webpack-merge');
const common = require('./webpack.common.js');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = merge(common, {
  mode: 'production',
  output: {
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist', 'js'),
  },
  plugins: [
    new CopyPlugin([
      { from: 'public', to: '..' },
    ]),
  ],
});
