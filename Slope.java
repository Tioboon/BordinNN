package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class Slope {

    InputBatch derivativeOutPut;
    InputBatch bVariableOutPut;

    public Slope(VectorResultNeuron actualResult, VectorResultNeuron nextResult){
        derivativeOutPut = CrossBatchesForSlope(actualResult, nextResult);
        bVariableOutPut = GetBByBatches(nextResult);
    }

    private InputBatch CrossBatchesForSlope(VectorResultNeuron actualResult, VectorResultNeuron nextResult){
        List<InputForNeurons> neuronInput = new ArrayList<>();
        for(int i = 0; i < actualResult.xOutPut.inputBatchList.size(); i++){
            List<Double> slopeList = new ArrayList<>();
            for (int j = 0; j < actualResult.xOutPut.inputBatchList.get(0).inputList.size(); j++){
                double deltaX =
                        nextResult.xOutPut.inputBatchList.get(i).inputList.get(j) -
                                actualResult.xOutPut.inputBatchList.get(i).inputList.get(j);
                double deltaY =
                        nextResult.yOutPut.inputBatchList.get(i).inputList.get(j) -
                                actualResult.yOutPut.inputBatchList.get(i).inputList.get(j);
                double slope = deltaY/deltaX;
                if(Double.isNaN(slope)){
                    slopeList.add(0d);
                    continue;
                }
                slopeList.add(slope);
            }
            neuronInput.add(new InputForNeurons(slopeList));
        }
        return new InputBatch(neuronInput);
    }

    //Get the ax ^ 2 {+ b} from non-linear function
    private InputBatch GetBByBatches(VectorResultNeuron nextResult){
        List<InputForNeurons> bValuesInput = new ArrayList<>();
        for (int i = 0; i < nextResult.xOutPut.inputBatchList.size(); i++) {
            List<Double> bValues = new ArrayList<>();
            for (int j = 0; j < nextResult.xOutPut.inputBatchList.get(0).inputList.size(); j++) {
                bValues.add(nextResult.yOutPut.inputBatchList.get(i).inputList.get(j)
                        - derivativeOutPut.inputBatchList.get(i).inputList.get(j)
                        * nextResult.xOutPut.inputBatchList.get(i).inputList.get(j));
            }
            bValuesInput.add(new InputForNeurons(bValues));
        }
        return new InputBatch(bValuesInput);
    }
}
