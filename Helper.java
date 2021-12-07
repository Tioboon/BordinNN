package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class Helper {

    //One hot is a list with 1 where the answer is correct and 0 where is wrong
    public InputBatch CalculateLoss(InputBatch softMaxedOutput, InputForNeurons neuronInput){
        List<InputForNeurons> outPut = new ArrayList<>();
        for (InputForNeurons input: softMaxedOutput.inputBatchList) {
            List<Double> finalDoubleList = new ArrayList<>();
            for (int i = 0; i < input.inputList.size(); i++) {
                finalDoubleList.add(Math.log(
                        input.inputList.get(i) * neuronInput.inputList.get(i)
                ));
            }
            outPut.add(new InputForNeurons(finalDoubleList));
        }
        return new InputBatch(outPut);
    }
}
