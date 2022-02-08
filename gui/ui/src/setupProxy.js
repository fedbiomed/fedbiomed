const { createProxyMiddleware } = require('http-proxy-middleware');

if (!process.env.NODE_ENV || process.env.NODE_ENV === 'development') {
    module.exports = function(app) {
      app.use(
        '/api',
        createProxyMiddleware({
          target: 'http://localhost:8484',
          changeOrigin: true,
        })
      );
    };
} else {
    console.log('Production mode is enabled')
}

