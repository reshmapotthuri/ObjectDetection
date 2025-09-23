using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using ObjectDetection.Models;
using SkiaSharp;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Tensorflow.Contexts;
using Tensorflow.Keras.Engine;

namespace ObjectDetection.Service
{
    public class ObjectDetectionService
    {
        public List<DetectionResult> Predict(string filePath)
        {
            var input = new List<ImageInput> { new ImageInput { ImagePath = filePath } };

            var _mlContext = new MLContext();

               var pipeline = _mlContext.Transforms
                .LoadImages(outputColumnName: "Image", imageFolder: "", inputColumnName: nameof(ImageInput.ImagePath))
                .Append(_mlContext.Transforms.ResizeImages("Image", 416, 416))
                .Append(_mlContext.Transforms.ExtractPixels("Image"))
                .Append(_mlContext.Transforms.CopyColumns("input_1:0", "Image")) // ✅ match ONNX input node name
                .Append(_mlContext.Transforms.ApplyOnnxModel(
                    modelFile: "wwwroot/models/object-detection.onnx",
                    outputColumnNames: new[] { "Identity:0", "Identity_1:0", "Identity_2:0" },
                    inputColumnNames: new[] { "input_1:0" })); // ✅ no ":0"

            var data = _mlContext.Data.LoadFromEnumerable(input);
            var _model = pipeline.Fit(data);
            var predictions = _model.Transform(data);

            var boxes = predictions.GetColumn<float[]>("Identity:0").ToArray();
            var labels = predictions.GetColumn<float[]>("Identity_1:0").ToArray();
            var scores = predictions.GetColumn<float[]>("Identity_2:0").ToArray();
            var labelMap = new Dictionary<int, string>
                {
                    { 936 , "person" },
                    { 1, "bicycle" },
                    { 2, "car" },
                    { 6036, "motorbike" },
                    { 4, "aeroplane" },
                    { 5, "bus" },
                    { 6, "train" },
                    { 7, "truck" },
                    { 8, "boat" },
                    { 9, "traffic light" },
                    // ... up to 80 classes for COCO dataset
                };
            string uploadedFileName = Path.GetFileName(filePath); // e.g., "image1.jpg"
            string relativePath = Path.Combine("uploads", uploadedFileName); // "~/uploads/image1.jpg"
            var results = new List<DetectionResult>();
            for (int i = 0; i < boxes.Length; i++)
            {
                Console.WriteLine($"Label length: {labels[i].Length}");
                var score = scores[i][0]; // Access first float in the score array
                if (score > 0.2)
                {
                      int labelId = labels[i]
                        .Select((score, index) => new { score, index })
                        .OrderByDescending(x => x.score)
                        .First().index;

                    var box = boxes[i];
                    results.Add(new DetectionResult
                    {
                        LabelId = labelId,
                        Confidence = score,
                        BoundingBox = new RectangleF(box[0], box[1], box[2], box[3]),
                        LabelName = labelMap.ContainsKey(labelId) ? labelMap[labelId] : labelId.ToString(),
                        ImagePath= relativePath

                    });
                }

            }

            return results;
        }


    }
}
