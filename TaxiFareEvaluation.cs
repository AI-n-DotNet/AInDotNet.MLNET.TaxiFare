using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AInDotNet.MLNET.TaxiFare;

public class TaxiFareEvaluation
{
    public string VendorId { get; set; } = string.Empty;
    public string RateCode { get; set; } = string.Empty;
    public float PassengerCount { get; set; }
    public float TripTime { get; set; }
    public float TripDistance { get; set; }
    public string PaymentType { get; set; } = string.Empty;
    public float FareAmount { get; set; }

    [ColumnName("Score")]
    public float PredictedFare { get; set; }
}