// Open links that leave the book in a new tab. Links between pages of the book stay in the same tab.
(function () {
    function markExternalLinks() {
        document.querySelectorAll('a[href]').forEach(function (a) {
            var url;
            try {
                url = new URL(a.getAttribute('href'), window.location.href);
            } catch (e) {
                return;
            }
            var isWeb = url.protocol === 'http:' || url.protocol === 'https:';
            if (isWeb && url.host !== window.location.host) {
                a.target = '_blank';
                a.rel = 'noopener noreferrer';
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', markExternalLinks);
    } else {
        markExternalLinks();
    }
})();
