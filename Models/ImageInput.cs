using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using SkiaSharp;
using System.Drawing;

namespace ObjectDetection.Models
{
    public class ImageInput
    {
        public string ImagePath { get; set; }
    }
    public class DetectionResult
    {
        public int LabelId { get; set; }
        public string LabelName { get; set; }
        public float Confidence { get; set; }
        public RectangleF BoundingBox { get; set; }
        public string ImagePath { get; set; }
    }

}
