var image;
var cropper;
var online = true;
window.addEventListener('DOMContentLoaded', function () { 
  var button = document.getElementById('crop-button'); 
  $.ajax({
      url: "http://ec2-54-219-178-149.us-west-1.compute.amazonaws.com:5000",
      success: function (response) { 
        button.disabled = false;
      },
      error: function (xhr, ajaxOptions, thrownError) { 
        online = false;
        button.disabled = true;
        window.alert("Cropping functionality offline :(. Give me free AWS credits?");
        console.log("yeah");
      },
      timeout: 500
  });
  var first = true;
  var canvas = document.getElementById('canvas');
  var ctx = canvas.getContext("2d");
  var outImage = document.getElementById('out_img');
  canvas.width = 226;
  canvas.height = 218;
  ctx.drawImage(outImage, 0, 0);

  image = document.getElementById('cropImg');
  cropper  = new Cropper(image, {
     autoCropArea:0.5,
     ready: function () { 
       if(first){
           cropper.setCropBoxData({'left': cropper.getContainerData().width*0.24, 
                                   'top': cropper.getContainerData().width*0.17, 
                                   'width': cropper.getContainerData().width*0.38, 
                                   'height': cropper.getContainerData().height*0.46 });
           first = false;
       }
     }
    });

  
  $(".thumbnail").click(function(event){
    var img = document.getElementById(event.target.id);
    image.src = img.src;
    cropper.replace(img.src);
  });

  var dc_url = 'http://ec2-54-219-178-149.us-west-1.compute.amazonaws.com:5000';
  var loader = document.getElementById('loader');
  $("#crop-button").click(function(event){
    if(!button.disabled){
      button.disabled = true;
      var encoded = cropper.getCroppedCanvas().toDataURL("image/jpeg");
      encoded = encoded.substring(encoded.indexOf(',')+1);
      canvas.style.display = "none";
      loader.style.display = "block";
      msg = {'img_64':encoded};
      $.ajax({
        type: "POST",
        url: dc_url,
        data: JSON.stringify(msg),
        success: function(data) {
          var image = new Image();
          loading = false;
          image.onload = function() {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);
            canvas.style.display = "";
            loader.style.display = "none";
            button.disabled = false;
          };
          image.src = "data:image/jpg;base64,"+data;
        },
        error: function() {
           window.alert("Something went wrong :( Try again?");
           canvas.style.display = "";
           loader.style.display = "none";
           button.disabled = false;
        },
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        timeout: 15000 
       });
     }
  }); 
});

function loadImg() {
  var file    = document.querySelector('input[type=file]').files[0];
  var reader  = new FileReader();

  reader.addEventListener("load", function () {
    image.src = reader.result;
    cropper.replace(image.src);
  }, false);
  if (file) {
    reader.readAsDataURL(file);
  }
}

   
