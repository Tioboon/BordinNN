package BordinNN;

import java.util.List;

public class NeuronIterator {

    public double lowestLoss = Double.POSITIVE_INFINITY;
    public HiddenLayers lowestLossNeuronsInfo;

    public NeuronIterator(double lowestLoss, HiddenLayers firstNeuronsInfo){
        this.lowestLoss = lowestLoss;
        lowestLossNeuronsInfo = firstNeuronsInfo;
    }

    public boolean CompareLoss(double loss){
        if(loss < lowestLoss){
            lowestLoss = loss;
            return true;
        }
        return false;
    }
}
