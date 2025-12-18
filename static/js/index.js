document.addEventListener('DOMContentLoaded', () => {
  // Carousel initialization
  if (typeof bulmaCarousel === 'function') {
    bulmaCarousel.attach('#carousel', {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      autoplay: true,
      autoplaySpeed: 3000,
    });
  }

  // Slider initialization
  if (typeof bulmaSlider === 'function') {
    bulmaSlider.attach('.slider', {
      start: 50,
      end: 100,
      step: 1,
    });
  }

  // Tooltip initialization
  if (typeof tippy === 'function') {
    tippy('.has-tooltip', {
      placement: 'top',
      arrow: true,
    });
  }
});
