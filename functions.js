// Add an event listener to the sidebar
document.getElementById('sidebar').addEventListener('click', function () {
  const sidebar = document.getElementById('sidebar');
  
  // Toggle active class
  sidebar.classList.toggle('active');
  
  // Dynamically change sidebar content
  if (sidebar.classList.contains('active')) {
    sidebar.innerHTML = "<p>Updated Sidebar Content</p>";
  } else {
    sidebar.innerHTML = "<p>Original Sidebar Content</p>";
  }
});

(function() {
    const sidebar = document.querySelector('aside[aria-expanded]');
    if (!sidebar) return;

    const observer = new MutationObserver(() => {
        const isOpen = sidebar.getAttribute('aria-expanded') === 'true';
        const event = new Event(isOpen ? 'open' : 'closed');
        document.dispatchEvent(event);
    });

    observer.observe(sidebar, { attributes: true });

    document.addEventListener('open', () => {
        window.parent.postMessage({ sidebarOpen: true }, "*");
    });

    document.addEventListener('closed', () => {
        window.parent.postMessage({ sidebarOpen: false }, "*");
    });
})();


let lastWidth = 0;
const detectSidebarState = () => {
    const sidebar = document.querySelector('[data-testid="stSidebarCollapsedControl"]');
    const sidebarWidth = sidebar ? sidebar.offsetWidth : 0;

    if (lastWidth !== sidebarWidth) {
        lastWidth = sidebarWidth;
        const event = new CustomEvent(sidebarWidth > 0 ? "sidebarOpen" : "sidebarClosed");
        window.dispatchEvent(event);
    }
};
setInterval(detectSidebarState, 300); // Check every 300ms

