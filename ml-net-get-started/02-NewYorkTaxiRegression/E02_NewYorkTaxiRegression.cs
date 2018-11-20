using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ml_net_get_started._02_NewYorkTaxiRegression
{
    class E02_NewYorkTaxiRegression : ITask
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "02-NewYorkTaxiRegression", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "02-NewYorkTaxiRegression", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "02-NewYorkTaxiRegression", "Model.zip");
        static TextLoader _textLoader;

        public void Run()
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                            {
                                new TextLoader.Column("VendorId", DataKind.Text, 0),
                                new TextLoader.Column("RateCode", DataKind.Text, 1),
                                new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                                new TextLoader.Column("TripTime", DataKind.R4, 3),
                                new TextLoader.Column("TripDistance", DataKind.R4, 4),
                                new TextLoader.Column("PaymentType", DataKind.Text, 5),
                                new TextLoader.Column("FareAmount", DataKind.R4, 6)
                            }
            }
            );

            var model = Train(mlContext, _trainDataPath);

            SaveModelAsFile(mlContext, model);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                    .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                    .Append(mlContext.Regression.Trainers.FastTree());

            Console.WriteLine("=============== Create and Train the Model ===============");

            var model = pipeline.Fit(dataView);

            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine($"*************************************************");

        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            //load the model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = loadedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            //Sample: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);

            Console.WriteLine("The model is saved to {0}", _modelPath);
        }


        public class TaxiTrip
        {
            [Column("0")]
            public string VendorId;

            [Column("1")]
            public string RateCode;

            [Column("2")]
            public float PassengerCount;

            [Column("3")]
            public float TripTime;

            [Column("4")]
            public float TripDistance;

            [Column("5")]
            public string PaymentType;

            [Column("6")]
            public float FareAmount;
        }

        public class TaxiTripFarePrediction
        {
            [ColumnName("Score")]
            public float FareAmount;
        }
    }
}
