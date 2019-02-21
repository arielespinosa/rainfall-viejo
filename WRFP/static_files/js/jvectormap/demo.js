
!function ($) {

  $(function(){

    $('#world_map').vectorMap({
      map: 'world_mill_en',
      normalizeFunction: 'polynomial',
      backgroundColor: '#25313e',     
      regionsSelectable: true,
      borderWidth: 5,
      zoomOnScroll: false,
      zoomButtons: false,
      
      series: {
        regions: [{
          attribute: 'fill'
        }]
      },

      regionStyle: {
        initial: {
          fill: '#36424f',
        },

        hover: {
          fill: '#65bd77',
          cursor: 'pointer'
        },

        selected: {
          fill: '#65bd77',
          cursor: 'pointer'
        },
      },
      markerStyle: {
        initial: {
          fill: '#65bd77',
          stroke: '#fff'
        }
      },

      onRegionSelected: function(){        
        //alert($('.jvectormap-region').attr('data-code')); 
       // alert(code); 
      },   
      
      onRegionClick: function(e, code) {            
        alert(code); 
      },    

    });

  });


}(window.jQuery);