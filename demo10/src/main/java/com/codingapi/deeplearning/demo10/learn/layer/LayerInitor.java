package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.core.InputType;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author lorne
 * @date 2020/2/11
 * @description
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class LayerInitor {


    private double lamdba;
    private double alpha;
    private int seed;
    private InputType inputType;


    @Override
    public String toString() {
        return "LayerInitor{" +
                ", inputType=" + inputType +
                '}';
    }
}
