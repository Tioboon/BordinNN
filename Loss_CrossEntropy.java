package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class Loss_CrossEntropy {

    public List<Double> resultBatch;
    public List<List<Double>> d;

    //Start backwards process
    public Loss_CrossEntropy(){}


    public List<Double> Forward(List<List<Double>> lastLayerOutput, List<List<Double>> trueBatch) {
        List<Double> lossBatch = new ArrayList<>();
        for (int i = 0; i < lastLayerOutput.size(); i++) {
            List<Double> loss = new ArrayList<>();
            for (int j = 0; j < lastLayerOutput.get(i).size(); j++) {
                double trueMatrixValue = trueBatch.get(i).get(j);
                double activationOutput = lastLayerOutput.get(i).get(j);
                double logActivation = Math.log(activationOutput);
                loss.add(trueMatrixValue * logActivation);
            }
            lossBatch.add(-MathHelper.SumVector(loss));
        }
        resultBatch = lossBatch;
        return lossBatch;
    }

    public List<List<Double>> Backward(List<List<Double>> dResult, List<List<Double>> trueBatch) {
        List<List<Double>> derivative = new ArrayList<>();
        for (int i = 0; i < trueBatch.size(); i++) {
            List<Double> dLoss = new ArrayList<>();
            for (int j = 0; j < trueBatch.get(i).size(); j++) {
                dLoss.add(-trueBatch.get(i).get(j) * dResult.get(i).get(j) / trueBatch.size());
            }
            derivative.add(dLoss);
        }
        d = derivative;
        return derivative;
    }
}
