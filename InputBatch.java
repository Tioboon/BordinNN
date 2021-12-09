package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class InputBatch {
    //Like a input list (To make the result value oscillate less when adding info to the "chart")
    public List<InputForNeurons> inputBatchList;

    public InputBatch(List<InputForNeurons> inputListForNeuron){
        inputBatchList = inputListForNeuron;
    }

}
