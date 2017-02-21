// augmentation

function demo(parent, width, height, data) 
{
  // canvas
  var canvas = parent.canvas;
  var ctx = canvas.getContext('2d');

  // parameters
  var datasetName;
  var summaryFile;
  var viewTopSamples = false;

  var mcw = 40;
  var mch = 40;
  var samples_grid_margin = 2;
  var sample_scale = 1.0;
  var selected = {a:2, p:2};

  // variables
  var data, net, classes, nc, dim;
  var cellsize, mx, my;
  var summary;

  function draw_confusion_matrix_box(cellsize){
    ctx.beginPath();
      ctx.fillStyle = 'rgba(255,255,255,1.0)';
      ctx.strokeStyle = 'rgba(0,0,0,1.0)';
      ctx.lineWidth = 1.0;
      ctx.fillRect(0, 0, nc * cellsize.x, nc * cellsize.y);
      ctx.rect(0, 0, nc * cellsize.x, nc * cellsize.y);
      ctx.stroke();
      ctx.closePath();
  };

  function draw_confusion_matrix_grid(cellsize){
      ctx.lineWidth = 0.75;
      ctx.strokeStyle = 'rgba(0,0,0,0.35)';
      for (var p=0; p<nc+1; p++) {
        ctx.beginPath();
        ctx.moveTo(0, p * cellsize.y);
        ctx.lineTo(nc * cellsize.x, p * cellsize.y);
        ctx.stroke();
        ctx.closePath();
      }
      for (var a=0; a<nc+1; a++) {
        ctx.beginPath();
        ctx.moveTo(a * cellsize.x, 0);
        ctx.lineTo(a * cellsize.x, nc * cellsize.y);
        ctx.stroke();
        ctx.closePath();
      }
  };

  function draw_confusion_matrix_labels(cellsize){
      ctx.font = '14px Arial';
      ctx.fillStyle = 'rgba(0,0,0,1.0)';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (var a=0; a<nc; a++) {
        ctx.fillText((datasetName=='MNIST'?'actual ':'')+classes[a], -5, (a + 0.5) * cellsize.y);
      }
      ctx.textAlign = 'left';
      for (var p=0; p<nc; p++) {
        ctx.save();
        ctx.translate((p + 0.5) * cellsize.x, 0);
        ctx.rotate(-0.5);
        ctx.fillText("predicted "+classes[p], -2, -12);
        ctx.restore();
      }
  };

  function get_color(pct) {
    if (pct < 0.08) {
      return 'hsl(240, 100%, ' + (20 + pct * 380) + '%)';
    }
    if (pct < 0.9) {
      return 'hsl(' + (263 - pct * 292) + ', 100%, 50%)';
    }
    return 'hsl(0, 100%, ' + (277 - pct * 250) + '%)';
  }
  function draw_confusion_matrix(x_, y_, cellsize, fontsize) 
  {
    function draw_cell(summary, a, p) {
      var count = summary.confusion[a][p];
      var pct = summary.actuals[a] > 0 ? summary.confusion[a][p] / summary.actuals[a] : 0;
          ctx.save();
          ctx.font = fontsize+'px Arial';
          ctx.textAlign = 'center';   
          ctx.textBaseline = 'middle';
          ctx.translate(p * cellsize.x, a * cellsize.y);
          //ctx.fillStyle = (a==p) ? 'rgba(0,255,0,'+pct+')' : 'rgba(255,0,0,'+pct+')';
          ctx.fillStyle = get_color(pct);
          ctx.fillRect(0, 0, cellsize.x, cellsize.y);
          ctx.fillStyle = (pct >= 0.18 && pct <= 0.8) ? 'rgba(0,0,0,1.0)' : 'rgba(255, 255, 255, 1.0)';
          ctx.fillText(count, cellsize.x/2.0, cellsize.y/2.0);
          ctx.restore();
      }
      
      ctx.save();
      ctx.translate(x_, y_);  

      draw_confusion_matrix_box(cellsize);
      for (var p=0; p<nc; p++) {
        for (var a=0; a<nc; a++) {
          draw_cell(summary, a, p);
        }
      }     
      draw_confusion_matrix_grid(cellsize);
    draw_confusion_matrix_labels(cellsize);

    // precision, recall, accuracy
      ctx.save();
      ctx.textAlign = 'right';
      ctx.fillText('precision', -5, nc * cellsize.y + 50);
      ctx.fillText('f-score', -5, nc * cellsize.y + 100);
      ctx.textAlign = 'left';
      ctx.translate(nc * cellsize.x + 48, -12);
      ctx.rotate(-0.5);
      ctx.fillText('recall', 0, 0);
      ctx.restore();
      ctx.strokeStyle = 'rgba(0,0,0,1.0)';
      ctx.textAlign = 'center';
      for (var a=0; a<nc; a++) {
        var recall = summary.actuals[a] == 0 ? 0 : summary.confusion[a][a] / summary.actuals[a];
        ctx.fillStyle = 'rgba('+Math.round(255*(1.0-recall))+','+Math.round(255*recall)+',0,1.0)';
        ctx.fillRect((nc - 0.5) * cellsize.x + 50, a * cellsize.y, cellsize.x, cellsize.y);
        ctx.rect((nc - 0.5) * cellsize.x + 50, a * cellsize.y, cellsize.x, cellsize.y);
        ctx.stroke();
        ctx.fillStyle = 'rgba(0,0,0,1.0)';
          ctx.fillText((100*recall).toFixed(1)+'%', nc * cellsize.x + 50, (a + 0.5) * cellsize.y);
      }     
      for (var p=0; p<nc; p++) {
        var precision = summary.predictions[p] == 0 ? 0 : summary.confusion[p][p] / summary.predictions[p];
        ctx.fillStyle = 'rgba('+Math.round(255*(1.0-precision))+','+Math.round(255*precision)+',0,1.0)';
        ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(p * cellsize.x, (nc - 0.5) * cellsize.y + 50, cellsize.x, cellsize.y);
        ctx.rect(p * cellsize.x, (nc - 0.5) * cellsize.y + 50, cellsize.x, cellsize.y);
        ctx.stroke();
        ctx.fillStyle = 'rgba(0,0,0,1.0)';
        ctx.fillText((100*precision).toFixed(1)+'%', (p + 0.5) * cellsize.x, nc * cellsize.y + 50);
      }
      for (var p=0; p<nc; p++) {
        var recall = summary.actuals[p] == 0 ? 0 : summary.confusion[p][p] / summary.actuals[p];
        var precision = summary.predictions[p] == 0 ? 0 : summary.confusion[p][p] / summary.predictions[p];
        var f = 2.0 * (precision * recall) / (precision + recall);
        ctx.fillStyle = 'rgba('+Math.round(255*(1.0-f))+','+Math.round(255*f)+',0,1.0)';
        ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(p * cellsize.x, (nc - 0.5) * cellsize.y + 100, cellsize.x, cellsize.y);
        ctx.rect(p * cellsize.x, (nc - 0.5) * cellsize.y + 100, cellsize.x, cellsize.y);
        ctx.stroke();
        ctx.fillStyle = 'rgba(0,0,0,1.0)';
        ctx.fillText((100*f).toFixed(1)+'%', (p + 0.5) * cellsize.x, nc * cellsize.y + 100);
      }     
      var accuracy = summary.correct / summary.total;
      ctx.fillStyle = 'rgba('+Math.round(255*(1.0-accuracy))+','+Math.round(255*accuracy)+',0,1.0)';
      ctx.fillRect((nc - 0.5 - 0.3) * cellsize.x + 50, (nc - 0.5 - 0.25) * cellsize.y + 70, cellsize.x * 1.6, cellsize.y * 1.5);
      ctx.rect((nc - 0.5 - 0.3) * cellsize.x + 50, (nc - 0.5 - 0.25) * cellsize.y + 70, cellsize.x * 1.6, cellsize.y * 1.5);
      ctx.stroke();
      ctx.fillStyle = 'rgba(0,0,0,1.0)';
      ctx.fillText('accuracy', nc * cellsize.x + 50, nc * cellsize.y + 70 - 10);
      ctx.fillText((accuracy*100).toFixed(2)+'%', nc * cellsize.x + 50, nc * cellsize.y + 70 + 10);
      ctx.restore();
  };

  function draw_confusion_matrix_samples(x_, y_, scale) 
  {
      function draw_confusion_matrix_samples_grid() {
        function draw_cell(summary, a, p) {
        var tops = summary.tops[a][p];
        if (tops.length == 0) return;
        ctx.fillStyle = (a==p) ? 'rgba(0,255,0,'+tops[0].prob+')' : 'rgba(255,0,0,'+tops[0].prob+')';
          ctx.fillRect(p * cellsize.x, a * cellsize.y, cellsize.x, cellsize.y);
          data.draw_sample(ctx, tops[0].idx, x_ + p * cellsize.x + samples_grid_margin, y_ + a * cellsize.y + samples_grid_margin, scale);
        }
        ctx.save();
        ctx.translate(x_, y_);  
        draw_confusion_matrix_box(cellsize);
      for (var p=0; p<nc; p++) {
          for (var a=0; a<nc; a++) {
            draw_cell(summary, a, p);
          }
        }       
        draw_confusion_matrix_grid(cellsize);
        draw_confusion_matrix_labels(cellsize);
        ctx.restore();
      };

      
    // get inexes of batches we are drawing from
    var batch_idxs = [];
    for (var p=0; p<nc; p++) {
        for (var a=0; a<nc; a++) {
          var tops = summary.tops[a][p];
          if (tops.length > 0) {
            batch_idxs.push(data.get_batch_idx_from_sample_idx(tops[0].idx));
          }
        }
    }

    batch_idxs = batch_idxs.filter(function(item, i, ar){ return ar.indexOf(item) === i; });
    data.load_multiple_batches(batch_idxs, draw_confusion_matrix_samples_grid);

  };

  function draw_confusion_samples(x_, y_, height, p, a, scale)
  {
    function draw_confusion_samples_box() {
      for (var i=0; i<Math.min(t.length, cols*rows); i++) {
        var c = i % cols;
        var r = Math.floor(i / cols);
        var x = x_ + margin + c * (dim * scale + margin);
        var y = y_ + margin + headerHeight + r * (dim * scale + margin + textHeight);

        // draw sample
        data.draw_sample(ctx, t[i].idx, x, y, scale);

        // prob
        ctx.font = fontSizePct+'px Arial';
        ctx.textAlign = 'center'
        ctx.fillStyle = 'rgba(0,0,0,1.0)';
        ctx.fillText((t[i].prob*100).toFixed(2)+'%', x + (dim * scale)/2.0, y + (dim * scale) + fontSizePct);
      }
    };

    var cols = 4;
    var margin = 8;
    var textHeight = 18;
    var headerHeight = 25;
    var fontSizePct = 12;
    var fontSizeHeader = 14;

    var width = margin + cols * (dim * scale + margin);
    var rows = Math.floor((height - headerHeight) / (dim * scale + textHeight));
    
    ctx.fillStyle = 'rgba(255,255,255,1.0)';
      ctx.strokeStyle = 'rgba(0,0,0,1.0)';
    ctx.fillRect(x_, y_, width, height);
    ctx.rect(x_, y_, width, height);
    ctx.stroke();
    ctx.font = fontSizeHeader+'px Arial';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(0,0,0,1.0)';
    ctx.fillText(classes[a] + (a==p?' correctly classified as ':' misclassified as ') + classes[p], x_+5, y_+20);
    
    // all the samples we need to draw
    var t = summary.tops[a][p];

    // get inexes of batches we are drawing rom
    var batch_idxs = [];
      for (var i=0; i<Math.min(t.length, cols*rows); i++) {
        batch_idxs.push(data.get_batch_idx_from_sample_idx(t[i].idx))
      }
    batch_idxs = batch_idxs.filter(function(item, i, ar){ return ar.indexOf(item) === i; });
    // load batches if necessary, then draw the samples
    data.load_multiple_batches(batch_idxs, draw_confusion_samples_box);
  };

  function update_canvas() {
    toggleView(viewTopSamples);
    ctx.fillStyle = 'rgba(255,255,255,1.0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    draw_confusion_samples(100 + nc*mcw + 120, 24, canvas.height-50, selected.p, selected.a, 2);
    if (viewTopSamples) {
      draw_confusion_matrix_samples(mx, my, sample_scale);
    } else {
      draw_confusion_matrix(mx, my, {x:mcw + 4, y:mch + 4}, 16);
    }
    //draw_confusion_samples(100 + nc*mcw + 100, 24, canvas.height-40, selected.p, selected.a, 2);
  };

  function test_all() {
    console.log("test "+numTest)
    net.test(numTest, update_canvas);
  };

  function test_individually() {
    update_canvas();
    setTimeout(function() {
      net.test(1, test_individually);   // when to stop?
    }, 100);
  };

  function mouseMoved(evt) {
    var canvas_rect = canvas.getBoundingClientRect();
    var mouse_x = evt.clientX - canvas_rect.left;
      var mouse_y = evt.clientY - canvas_rect.top;
    var mx_ = Math.floor((mouse_x - mx) / cellsize.x);
    var my_ = Math.floor((mouse_y - my) / cellsize.y);
    if (mx_ >= 0 && mx_ < nc && my_ >= 0 && my_ <nc &&
      (mx_ != selected.p || my_ != selected.a)) {
      selected = {a: my_, p: mx_};    
      update_canvas();
    }
  };

  function toggleView(viewTopSamples_) {
    viewTopSamples = viewTopSamples_;
      cellsize = {x:mcw + 4, y:mch + 4};
      mx = 100;
      my = 90;
    
  };

  add_control_panel_menu(["View numbers","View top samples"], function() {
    viewTopSamples = (this.value == "View top samples");
    update_canvas();
  });
    
  classes = data.get_classes();
  nc = classes.length;
  dim = data.get_dim();
  summaryFile = data.get_summary_file();
  $.getJSON(summaryFile, function(res) {
    summary = res;
    update_canvas();
  });

  canvas.addEventListener("mousemove", mouseMoved, false);
};

