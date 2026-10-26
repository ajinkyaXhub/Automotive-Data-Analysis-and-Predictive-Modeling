/**
 * Carlytics — Main JavaScript
 * Micro-interactions, number counters, and UI helpers
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log(
        '%c Carlytics ',
        'background: #4f46e5; color: #ffffff; font-weight: bold; padding: 4px 8px; border-radius: 4px;',
        'Machine Learning & Automotive Data Analytics Platform'
    );

    // Check if user prefers reduced motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (!prefersReducedMotion) {
        // Run count-up animation on numeric stat-value elements
        initStatCountUp();
    }
});

/**
 * Animate numeric values on stat cards counting up smoothly
 */
function initStatCountUp() {
    const statElements = document.querySelectorAll('.stat-value');
    
    statElements.forEach(el => {
        const text = el.textContent.trim();
        // Check if pure integer (e.g. 3577)
        if (/^\d+$/.test(text)) {
            const target = parseInt(text, 10);
            animateNumber(el, 0, target, 800, n => n.toLocaleString('en-IN'));
        }
    });
}

/**
 * Easing number counter helper using requestAnimationFrame
 */
function animateNumber(element, start, end, duration, formatFn) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // easeOutQuad curve
        const ease = 1 - (1 - progress) * (1 - progress);
        const currentVal = Math.round(start + (end - start) * ease);

        element.textContent = formatFn ? formatFn(currentVal) : currentVal;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = formatFn ? formatFn(end) : end;
        }
    }

    requestAnimationFrame(update);
}
