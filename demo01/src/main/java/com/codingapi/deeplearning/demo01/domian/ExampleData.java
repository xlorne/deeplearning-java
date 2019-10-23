package com.codingapi.deeplearning.demo01.domian;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;

/**
 * @author lorne
 * @date 2019-10-22
 * @description
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ExampleData {

    private BigDecimal x;
    private BigDecimal y;

}
