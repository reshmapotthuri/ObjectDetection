using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.ML;
using ObjectDetection.Models;
using ObjectDetection.Service;
using SkiaSharp;
using System;
using System.Drawing;
using Tensorflow.Contexts;
using Tensorflow.Keras.Engine;


namespace ObjectDetection.Pages
{
    public class IndexModel : PageModel
    {
        private readonly ILogger<IndexModel> _logger;
        private readonly ObjectDetectionService _ObjectDetectionService;
        public IndexModel(ILogger<IndexModel> logger,ObjectDetectionService ObjectDetectionService)
        {
            _logger = logger;
            _ObjectDetectionService = ObjectDetectionService;
        }

        public void OnGet()
        {

        }
        public async Task<IActionResult> OnPostAsync(IFormFile imageFile)
        {
            if (imageFile == null || imageFile.Length == 0)
                return Page();

            var uploadFolder = Path.Combine("wwwroot", "uploads");
            Directory.CreateDirectory(uploadFolder);
            var filePath = Path.Combine(uploadFolder, Guid.NewGuid() + ".png");

            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await imageFile.CopyToAsync(stream);
            }

           var results = _ObjectDetectionService.Predict(filePath);
 
            ViewData["Detections"] = results;
            return Page();
        }

    }
}
