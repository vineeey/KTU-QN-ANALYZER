/**
 * PYQ Analyzer - Main JavaScript
 * Enhanced with scroll animations and 3D effects
 */

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    // Initialize scroll reveal animations
    initScrollReveal();
    
    // Initialize 3D tilt effects
    init3DTilt();
    
    // Initialize smooth scrolling
    initSmoothScroll();
    
    // Re-initialize icons after HTMX swaps
    document.body.addEventListener('htmx:afterSwap', function() {
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
        // Re-initialize scroll reveal for new content
        initScrollReveal();
    });
});

// HTMX configuration
document.body.addEventListener('htmx:configRequest', function(evt) {
    // Add CSRF token to all HTMX requests
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]');
    if (csrfToken) {
        evt.detail.headers['X-CSRFToken'] = csrfToken.value;
    }
});

// Configuration constants
const SCROLL_REVEAL_CONFIG = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const TILT_3D_CONFIG = {
    perspective: 1000,
    maxRotation: 10,
    scale: 1.05
};

// Scroll Reveal Animation
function initScrollReveal() {
    const revealElements = document.querySelectorAll('.scroll-reveal');
    
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, SCROLL_REVEAL_CONFIG);
    
    revealElements.forEach(el => revealObserver.observe(el));
}

// 3D Tilt Effect for Cards
function init3DTilt() {
    const cards = document.querySelectorAll('.card-3d-tilt');
    
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / TILT_3D_CONFIG.maxRotation;
            const rotateY = (centerX - x) / TILT_3D_CONFIG.maxRotation;
            
            card.style.transform = `perspective(${TILT_3D_CONFIG.perspective}px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(${TILT_3D_CONFIG.scale}, ${TILT_3D_CONFIG.scale}, ${TILT_3D_CONFIG.scale})`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = `perspective(${TILT_3D_CONFIG.perspective}px) rotateX(0) rotateY(0) scale3d(1, 1, 1)`;
        });
    });
}

// Smooth Scrolling
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href !== '#') {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

// File upload preview
function previewFile(input, previewId) {
    const preview = document.getElementById(previewId);
    if (input.files && input.files[0]) {
        const file = input.files[0];
        preview.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Loading Animation
function showLoading(message = 'Loading...') {
    const loading = document.createElement('div');
    loading.id = 'loading-overlay';
    loading.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    loading.innerHTML = `
        <div class="bg-white dark:bg-gray-800 rounded-lg p-8 flex flex-col items-center space-y-4 animate-scale-in">
            <div class="w-16 h-16 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
            <p class="text-gray-700 dark:text-gray-300 font-medium">${message}</p>
        </div>
    `;
    document.body.appendChild(loading);
}

function hideLoading() {
    const loading = document.getElementById('loading-overlay');
    if (loading) {
        loading.classList.add('animate-fade-out');
        setTimeout(() => loading.remove(), 300);
    }
}

// Chart.js default configuration
if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = 'system-ui, -apple-system, sans-serif';
    Chart.defaults.plugins.legend.position = 'bottom';
    Chart.defaults.animation.duration = 1000;
    Chart.defaults.animation.easing = 'easeInOutQuart';
}

// Parallax Effect for Background
function initParallax() {
    const parallaxElements = document.querySelectorAll('.parallax');
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        
        parallaxElements.forEach(el => {
            const speed = el.dataset.speed || 0.5;
            el.style.transform = `translateY(${scrolled * speed}px)`;
        });
    });
}

// Initialize parallax if elements exist
if (document.querySelector('.parallax')) {
    initParallax();
}
