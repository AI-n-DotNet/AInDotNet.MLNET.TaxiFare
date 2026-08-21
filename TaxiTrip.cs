using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace AInDotNet.MLNET.TaxiFare;

public class TaxiTrip
{
    [LoadColumn(0)]
    public string VendorId { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string RateCode { get; set; } = string.Empty;

    [LoadColumn(2)]
    public float PassengerCount { get; set; }

    [LoadColumn(3)]
    public float TripTime { get; set; }

    [LoadColumn(4)]
    public float TripDistance { get; set; }

    [LoadColumn(5)]
    public string PaymentType { get; set; } = string.Empty;

    [LoadColumn(6)]
    public float FareAmount { get; set; }
}