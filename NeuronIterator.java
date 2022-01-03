package BordinNN;

import java.util.List;

public class NeuronIterator {

    public double lowestLoss = Double.POSITIVE_INFINITY;
    public double lowestAcc = 0d;
    public List<LayerDense> lowestLossNeuronsInfo;

    public boolean CompareLoss(double loss, List<LayerDense> lowestLossNeuronsInfo){
        if(loss < lowestLoss){
            lowestLoss = loss;
            this.lowestLossNeuronsInfo = lowestLossNeuronsInfo;
            return true;
        }
        return false;
    }

    public boolean CompareAccuracy(double acc, List<LayerDense> lowestLossNeuronsInfo){
        if(acc > lowestAcc){
            lowestAcc = acc;
            this.lowestLossNeuronsInfo = lowestLossNeuronsInfo;
            return true;
        }
        return false;
    }
}
