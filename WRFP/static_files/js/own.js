$(document).ready(function () {  
   //--Vars---------------------------------------------------------
   var titulo = document.title;
   
   //var ctx = document.getElementById("myChart").getContext('2d');
   //PlotLineChart(ctx);

   if((titulo.indexOf("-") == -1) && (titulo.indexOf("|") == -1)){
     var ctx2 = document.getElementById('canvas').getContext('2d');
     PlotMultiLineChart(ctx2);
   };



   // NO TOCAR DE AKI PARA ABAJO
   //-- Cambiar el formato del contenido de vertical a horizontal ----
   if(titulo.indexOf("-") != -1)
     $('#body').removeClass('vbox').addClass('hbox stretch');

    //--Notifications----------------------------------------------------------
    html1 = '<article class="media">' +
    '<span class="pull-left thumb-sm" style="padding-top: 9%;"><i class="fa fa-bell-o fa-3x"></i></span>'+                
    '<div class="media-body">' +
    '<div class="pull-right media-xs text-center text-muted">' +
    '<br>' +
    '</div>' +
    '<small class="block m-t-sm">Esto es una notificacion.</small>' +   
    '</div></article>';
     
    $('#index').on('click', function () {
		$.amaran({
        'theme'     :'colorful',
        'content'   :{		
           bgcolor:'#65bd77',
           color:'#fff',
           message: html1
        },
		'cssanimationIn'    :'bounceInRight',
        'cssanimationOut'   :'slideBottom',
        'position'  		:'bottom right',
		'sticky'        :true,
        'closeOnClick'  :false,
		'closeButton'   :true
        });   
        
        $('.jvector').on('click', function () {
           alert($(event.target));
        });    
       
    });

    
    //--Active/Desactive Users-------------------------------------
    /*$('#bla').on('clik', function () {        
        $.POST('/configruation/users/toggle_status', {'user_status':'0'}, function (response) {
            alert(response['status']);
        })

    });*/

    $('i').on('click', function () {    
        var element = $(event.target);
        $.getJSON('/configuration/users/toggle_status', {'user-status': $(this).attr('id')[0]}, function (response) {
               
           if(element.attr('class').indexOf('text-success') != -1){
                element.removeClass('text-success').addClass('text-danger');
                element.attr('title','Desactivar');

            }else{              
                element.removeClass('text-danger').addClass('text-success');
                element.attr('title','Activar');
            }
        });  
    });  
    //-------------------------------------------------------------------------

    //--Get Investigator-----------------------------------------------------------------------
    $('a').on('click', function () {    
        var element = $(event.target);     
        if(element.attr('id').indexOf('inv-') != -1) 
            $.get('/configuration/users/get_investigator', {'inv_pk': $(this).attr('id')[4]}, function (response) {
               $('#inv-name').html(response.investigator['name']);
            });                    
            /*$.getJSON('/configuration/users/get_investigator', {'inv_pk': $(this).attr('id')[4]}, function (response) {
                
                $('#inv-name').html = response.investigator.FullName
               
              if(response.investigator != 'Null'){                 
                }else{                        
                }
            }); */
    });  
      

     

    //--Show profile form------------------------------------------------------

 });			 