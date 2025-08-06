using Microsoft.ML;
using Microsoft.ML.Data;

public class EmotionData
{
    [LoadColumn(0)]
    public string Text;

    [LoadColumn(1)]
    public string Label;
}

public class EmotionPrediction
{
    [ColumnName("PredictedLabel")]
    public string Prediction;

    public float[] Score;
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        // Load data
        var data = mlContext.Data.LoadFromTextFile<EmotionData>(
            path: "C:\\Users\\dawit\\source\\repos\\dawitde2\\WellnessPulse\\WellnessPulse\\emotions.csv", hasHeader: true, separatorChar: ',');

        // Split data
        var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        // Build pipeline
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
            .Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(EmotionData.Text)))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Train model
        var model = pipeline.Fit(splitData.TrainSet);

        // Evaluate
        var predictions = model.Transform(splitData.TestSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
        Console.WriteLine($"\nModel Accuracy: {metrics.MicroAccuracy:P2}");

        // Prediction engine
        var predictor = mlContext.Model.CreatePredictionEngine<EmotionData, EmotionPrediction>(model);

        // Test loop
        while (true)
        {
            Console.Write("\nType a thought or feeling (or 'exit'): ");
            string input = Console.ReadLine();

            if (input?.ToLower() == "exit") break;

            var prediction = predictor.Predict(new EmotionData { Text = input });

            Console.WriteLine($"Predicted Emotion: {prediction.Prediction}");
        }
    }
}
