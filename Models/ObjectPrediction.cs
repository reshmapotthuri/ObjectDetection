using System.Drawing;

namespace ObjectDetection.Models
{
    public class ObjectPrediction
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
        public RectangleF BoundingBox { get; set; }

    }
}
