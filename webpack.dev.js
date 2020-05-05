const path = require('path');
const merge = require('webpack-merge');
const common = require('./webpack.common.js');

module.exports = merge(common, {
  mode: 'development',
  devtool: 'inline-source-map',
  devServer: {
    https: true,
    disableHostCheck: true,
    host: '0.0.0.0',
    hot: true,
    inline: true,
    historyApiFallback: true,
    open: true,
    openPage: 'index.html',
    contentBase: path.join(__dirname, 'public'),
    watchContentBase: true,
  },
});