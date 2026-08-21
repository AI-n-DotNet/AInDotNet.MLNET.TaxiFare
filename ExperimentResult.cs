namespace AInDotNet.MLNET.TaxiFare;

public class ExperimentResult
{
    public string Run { get; set; } = string.Empty;
    public string DataSet { get; set; } = string.Empty;
    public string Features { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;

    public double RSquared { get; set; }
    public double RMSE { get; set; }
    public double MAE { get; set; }
    public double MSE { get; set; }
}