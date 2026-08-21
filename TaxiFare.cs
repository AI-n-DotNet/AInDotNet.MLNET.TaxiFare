using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML;

namespace AInDotNet.MLNET.TaxiFare;

public class TaxiFare
{
    private readonly string inputDataFile = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "Data", "taxi-fare-train.csv"));
    private readonly MLContext mlContext = new(seed: 1);
    private readonly List<ExperimentResult> results = new();

    public void RunExperiments()
    {
        Console.WriteLine("============================================================");
        Console.WriteLine(" ML.NET TAXI FARE PREDICTION EXPERIMENT");
        Console.WriteLine("============================================================");
        Console.WriteLine();

        Console.WriteLine($"Framework: {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}");
        Console.WriteLine();

        if (!File.Exists(inputDataFile))
        {
            Console.WriteLine($"Training data file not found: {inputDataFile}");
            Console.WriteLine("See Data/README.md for download instructions.");
            return;
        }

        var data = mlContext.Data.LoadFromTextFile<TaxiTrip>(inputDataFile, hasHeader: true, separatorChar: ',');
        var rows = mlContext.Data.CreateEnumerable<TaxiTrip>(data, reuseRowObject: false).ToList();

        PrintDataProfile(rows);

        // Simple cleaning rule for this exercise:
        // remove trips with non-positive distance, duration, or fare.
        // This is intentionally conservative so students can experiment with
        // additional anomaly-detection and business-rule cleaning themselves.
        var filteredRows = rows.Where(x => x.TripDistance > 0 && x.TripTime > 0 && x.FareAmount > 0).ToList();
        var cleanData = mlContext.Data.LoadFromEnumerable(filteredRows);

        Console.WriteLine();
        Console.WriteLine($"Original rows: {rows.Count:N0}");
        Console.WriteLine($"Clean rows:    {filteredRows.Count:N0}");
        Console.WriteLine($"Rows removed:  {rows.Count - filteredRows.Count:N0}");
        Console.WriteLine($"Removed:       {(double)(rows.Count - filteredRows.Count) / rows.Count:P2}");
        Console.WriteLine();

        // Important: A and B share exactly the same split.
        var originalSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.20, seed: 123);

        // Important: C, D, E and F share exactly the same clean-data split.
        var cleanSplit = mlContext.Data.TrainTestSplit(cleanData, testFraction: 0.20, seed: 123);

        // Basic features:
        // PassengerCount, TripTime, TripDistance, PaymentType
        //
        // Full features:
        // Basic features + VendorId + RateCode
        RunExperiment("A", "Original", "Basic", "FastTree", originalSplit.TrainSet, originalSplit.TestSet);
        RunExperiment("B", "Original", "Full", "FastTree", originalSplit.TrainSet, originalSplit.TestSet);

        RunExperiment("C", "Clean", "Basic", "FastTree", cleanSplit.TrainSet, cleanSplit.TestSet);
        RunExperiment("D", "Clean", "Full", "FastTree", cleanSplit.TrainSet, cleanSplit.TestSet);

        RunExperiment("E", "Clean", "Full", "LightGBM", cleanSplit.TrainSet, cleanSplit.TestSet);
        RunExperiment("F", "Clean", "Full", "FastForest", cleanSplit.TrainSet, cleanSplit.TestSet);

        PrintComparison();
    }


    private void RunExperiment(string run, string dataType, string featureType, string algorithm, IDataView trainingData, IDataView testData)
    {
        Console.WriteLine();
        Console.WriteLine("========================================================================================================================");
        Console.WriteLine($"RUN {run}: {dataType} Data + {featureType} Features + {algorithm}");
        Console.WriteLine("========================================================================================================================");

        bool useFullFeatures = featureType == "Full";

        var pipeline = BuildPipeline(useFullFeatures, algorithm);

        Console.WriteLine("Training...");
        var model = pipeline.Fit(trainingData);

        Console.WriteLine("Evaluating...");
        var predictions = model.Transform(testData);

        var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(TaxiTrip.FareAmount));

        Console.WriteLine();
        Console.WriteLine($"R-Squared: {metrics.RSquared:0.0000}");
        Console.WriteLine($"RMSE:      {metrics.RootMeanSquaredError:0.0000}");
        Console.WriteLine($"MAE:       {metrics.MeanAbsoluteError:0.0000}");
        Console.WriteLine($"MSE:       {metrics.MeanSquaredError:0.0000}");

        results.Add(new ExperimentResult
        {
            Run = run,
            DataSet = dataType,
            Features = featureType,
            Algorithm = algorithm,
            RSquared = metrics.RSquared,
            RMSE = metrics.RootMeanSquaredError,
            MAE = metrics.MeanAbsoluteError,
            MSE = metrics.MeanSquaredError
        });

        PrintWorstPredictions(predictions, 20);
    }


    private IEstimator<ITransformer> BuildPipeline(bool useFullFeatures, string algorithm)
    {
        IEstimator<ITransformer> pipeline = mlContext.Transforms.Categorical.OneHotEncoding("PaymentTypeEncoded", nameof(TaxiTrip.PaymentType));

        if (useFullFeatures)
        {
            pipeline = pipeline.Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorIdEncoded", nameof(TaxiTrip.VendorId)));
            pipeline = pipeline.Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCodeEncoded", nameof(TaxiTrip.RateCode)));

            pipeline = pipeline.Append(mlContext.Transforms.Concatenate("Features",
                nameof(TaxiTrip.PassengerCount), nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance),
                "PaymentTypeEncoded", "VendorIdEncoded", "RateCodeEncoded"));
        }
        else
        {
            pipeline = pipeline.Append(mlContext.Transforms.Concatenate("Features",
                nameof(TaxiTrip.PassengerCount), nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance),
                "PaymentTypeEncoded"));
        }

        return algorithm switch
        {
            "FastTree" => pipeline.Append(mlContext.Regression.Trainers.FastTree(labelColumnName: nameof(TaxiTrip.FareAmount), featureColumnName: "Features")),
            "LightGBM" => pipeline.Append(mlContext.Regression.Trainers.LightGbm(labelColumnName: nameof(TaxiTrip.FareAmount), featureColumnName: "Features")),
            "FastForest" => pipeline.Append(mlContext.Regression.Trainers.FastForest(labelColumnName: nameof(TaxiTrip.FareAmount), featureColumnName: "Features")),
            _ => throw new ArgumentException($"Unknown algorithm: {algorithm}")
        };
    }


    private void PrintWorstPredictions(IDataView predictions, int count)
    {
        var worstPredictions = mlContext.Data.CreateEnumerable<TaxiFareEvaluation>(predictions, reuseRowObject: false)
            .Select(x => new
            {
                x.VendorId,
                x.RateCode,
                x.PassengerCount,
                x.TripTime,
                x.TripDistance,
                x.PaymentType,
                Actual = x.FareAmount,
                Predicted = x.PredictedFare,
                Error = Math.Abs(x.FareAmount - x.PredictedFare)
            })
            .OrderByDescending(x => x.Error)
            .Take(count)
            .ToList();

        Console.WriteLine();
        Console.WriteLine($"{count} Worst Predictions");

        Console.WriteLine($"{"Actual",10} {"Predicted",12} {"Error",10} {"Distance",10} {"Time",8} {"Pass",6} {"Payment",8} {"Rate",8} {"Vendor",8}");
        Console.WriteLine(new string('-', 100));

        foreach (var row in worstPredictions)
        {
            Console.WriteLine($"{row.Actual,10:C2} {row.Predicted,12:C2} {row.Error,10:C2} {row.TripDistance,10:F2} {row.TripTime,8:F0} {row.PassengerCount,6:F0} {row.PaymentType,8} {row.RateCode,8} {row.VendorId,8}");
        }
    }


    private void PrintDataProfile(List<TaxiTrip> rows)
    {
        Console.WriteLine("ORIGINAL DATA PROFILE");
        Console.WriteLine("---------------------");

        Console.WriteLine($"Total rows: {rows.Count:N0}");
        Console.WriteLine($"Zero distance: {rows.Count(x => x.TripDistance <= 0):N0}");
        Console.WriteLine($"Zero time: {rows.Count(x => x.TripTime <= 0):N0}");
        Console.WriteLine($"Zero distance AND zero time: {rows.Count(x => x.TripDistance <= 0 && x.TripTime <= 0):N0}");
        Console.WriteLine($"Fare > $100: {rows.Count(x => x.FareAmount > 100):N0}");
        Console.WriteLine($"Fare > $100 with zero distance: {rows.Count(x => x.FareAmount > 100 && x.TripDistance <= 0):N0}");

        var fares = rows.Select(x => x.FareAmount).OrderBy(x => x).ToList();

        Console.WriteLine();
        Console.WriteLine($"Minimum Fare:        {fares.First():C2}");
        Console.WriteLine($"Maximum Fare:        {fares.Last():C2}");
        Console.WriteLine($"Median Fare:         {Percentile(fares, 0.50):C2}");
        Console.WriteLine($"95th Percentile:     {Percentile(fares, 0.95):C2}");
        Console.WriteLine($"99th Percentile:     {Percentile(fares, 0.99):C2}");
    }


    private static float Percentile(List<float> sortedValues, double percentile)
    {
        int index = (int)Math.Round(percentile * (sortedValues.Count - 1));
        return sortedValues[index];
    }


    private void PrintComparison()
    {
        Console.WriteLine();
        Console.WriteLine();
        Console.WriteLine("========================================================================================================================");
        Console.WriteLine("FINAL EXPERIMENT COMPARISON");
        Console.WriteLine("========================================================================================================================");

        Console.WriteLine();
        Console.WriteLine($"{"Run",-5} {"Data",-12} {"Features",-12} {"Algorithm",-14} {"R-Squared",12} {"MAE",12} {"RMSE",12} {"MSE",12}");
        Console.WriteLine(new string('-', 95));

        foreach (var result in results)
        {
            Console.WriteLine($"{result.Run,-5} {result.DataSet,-12} {result.Features,-12} {result.Algorithm,-14} {result.RSquared,12:0.0000} {result.MAE,12:0.0000} {result.RMSE,12:0.0000} {result.MSE,12:0.0000}");
        }

        Console.WriteLine();
        Console.WriteLine("EXPERIMENT QUESTIONS");
        Console.WriteLine("--------------------");
        Console.WriteLine("A -> B: How much did adding business context / additional features help?");
        Console.WriteLine("A -> C: How much did cleaning the data help?");
        Console.WriteLine("C -> D: How much did additional features help after cleaning?");
        Console.WriteLine("D -> E: Does LightGBM outperform FastTree with identical data and features?");
        Console.WriteLine("D -> F: Does FastForest outperform FastTree with identical data and features?");
        Console.WriteLine("E -> F: Which alternative tree algorithm performs best?");
        Console.WriteLine();
    }
}