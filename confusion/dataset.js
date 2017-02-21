function dataset(_name, _summaryFile, _sw, _channels, _samplesPerBatch, _nBatches, _batchPath, _classes) 
{
	// initialize
	var self = this;
	var batchPath, idxBatch;
	var sw, sh, channels, samplesPerBatch, nBatches, columns;
	var classes;
	var idxSample;

	// setup canvases
	var batchImg;
	var batches;
	var batchCtx;
	var currentBatchData;
	var lastBatch;
	var summaryFile;

	this.get_name = function(){return datasetName;}
	this.get_dim = function(){return sw;}
	this.get_channels = function(){return channels;}
	this.get_samples_per_batch = function(){return samplesPerBatch;}
	this.get_batch_idx = function(){return idxBatch;}
	this.get_sample_idx = function(){return idxSample;}
	this.get_classes = function(){return classes;}
	this.get_summary_file = function(){return summaryFile;}

	function initialize() {
		idxBatch = -1;
		idxSample = 0;
		lastBatch = -1;
		batchImg = new Image();
		batches = [...Array(nBatches)];
		batchCtx = [...Array(nBatches)];
	};

	this.load_batch = function(idxBatch_, callback) {
		idxBatch = idxBatch_;
		if (batchCtx[idxBatch] === undefined) {
			batchImg.onload = function() {
				console.log("loaded "+ self.datasetName+" batch "+idxBatch);
				batches[idxBatch] = document.createElement('canvas');
				batches[idxBatch].width = sw * sh;
				batches[idxBatch].height = samplesPerBatch;   
				batchCtx[idxBatch] = batches[idxBatch].getContext('2d');
				batchCtx[idxBatch].drawImage(batchImg, 0, 0);
				callback();
			};
			batchImg.src = batchPath+"_batch_"+idxBatch+".png";
		} else {
			callback();
		}
	};

	this.load_next_batch = function(callback) {
		this.load_batch(idxBatch+1, callback);
	};

	this.load_multiple_batches = function(idxBatches, callback) {
		function load_next_batch_from_sequence() {
			if (idxBatches.length == 0) {
				callback();
			} else {
				var idxBatch_ = idxBatches.splice(0, 1);
				self.load_batch(idxBatch_, load_next_batch_from_sequence);
			}
		};
		load_next_batch_from_sequence();
	};

	this.get_batch_idx_from_sample_idx = function(idxSample) { 
		return Math.floor(idxSample / samplesPerBatch);
	};

	this.draw_current_sample = function(ctx, x, y, scale, grid_thickness, crop) {
		this.draw_sample(ctx, idxSample, x, y, scale, grid_thickness, crop);
	};

	this.get_sample_image = function(idx, callback) {
		var b = Math.floor(idx / samplesPerBatch);
		var k = idx % samplesPerBatch;
		if (batchCtx[b] === undefined) {
			this.load_batch(b, function() {
				var sample = batchCtx[b].getImageData(0, k, sw*sh, 1);
				callback({data:sample.data, sw:sw, sh:sh, channels:channels});
			})
		}
		else {
			var sample = batchCtx[b].getImageData(0, k, sw*sh, 1);
			callback({data:sample.data, sw:sw, sh:sh, channels:channels});
		}
	};

	this.draw_sample = function(ctx, idx, x, y, scale, grid_thickness, crop) {
		var sampleImg = this.get_sample_image(idx, function(sampleImg){
			var crop_ = (crop === undefined) ? {x:0, y:0, w:sw, h:sh, pad:0} : crop;
			var g = (grid_thickness === undefined) ? 0 : grid_thickness;
			var ny = crop_.h;
			var nx = crop_.w;
			var newImg = ctx.createImageData(nx * (scale + g), ny * (scale + g));
			for (var j=0; j<ny; j++) {
			 	for (var i=0; i<nx; i++) {
					var y_ = crop_.y + j - crop_.pad;
					var x_ = crop_.x + i - crop_.pad;
					var idxS = (y_ * sw + x_) * 4;
					if (y_ < 0 || y_ >= sh || x_ < 0 || x_ >= sw) {
						idxS = -1;	// in the padding
					}
					for (var sj=0; sj<scale+g; sj++) {
			      		for (var si=0; si<scale+g; si++) {
							var idxN = ((j * (scale + g) + sj) * nx * (scale + g) + (i * (scale + g) + si)) * 4;
			      			if (si < scale && sj < scale) {
				        		newImg.data[idxN  ] = idxS == -1 ? 0   : sampleImg.data[idxS  ];
				        		newImg.data[idxN+1] = idxS == -1 ? 0   : sampleImg.data[idxS+1];
				        		newImg.data[idxN+2] = idxS == -1 ? 0   : sampleImg.data[idxS+2];
				        		newImg.data[idxN+3] = idxS == -1 ? 255 : sampleImg.data[idxS+3];                						
				        	} else {
				        		newImg.data[idxN  ] = 127;
				        		newImg.data[idxN+1] = 127;
				        		newImg.data[idxN+2] = 127;
				        		newImg.data[idxN+3] = 255;
				        	}
			      		}
			    	}
			  	}
			}
			ctx.putImageData(newImg, x, y);
		});
	};

  summaryFile = _summaryFile;
	sw = sh = _sw;
  channels = _channels;
  samplesPerBatch = _samplesPerBatch;
  nBatches = _nBatches;
  batchPath = _batchPath;
  classes = _classes;
  initialize();
};
