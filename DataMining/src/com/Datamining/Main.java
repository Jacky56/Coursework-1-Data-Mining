package com.Datamining;

import java.util.*;
import org.jblas.*;

public class Main {
	public static void main(String[] args) {
		
		ArrayList<String[]> dataSet = DataManager.ReadCSV("data/adult.train.5fold.csv",false);
		
		DataManager.SetRegex("?");
		
		dataSet = DataManager.ReplaceMissingValues(dataSet);
		
		DoubleMatrix M = DataManager.dataSetToMatrix(dataSet);
		
		DoubleMatrix Xn = DataManager.Normalize(DataManager.GetFeatures(M, new int[]{14,15}));
		
		
		List<ArrayList<DoubleMatrix>> bins = DataManager.split(M, 15);
		
		//System.out.println(DataManager.GetClass(M,14));
		
		//System.out.println(Xn);
		
		System.out.println(bins.get(1).size());
		
		//--- debug: test if ? get replaced by common
//		System.out.println(dataSet.get(18)[13]);
//		dataSet = DataManager.RefactorDataSet(dataSet,"?");
//		System.out.println(dataSet.get(18)[13]);
		
		
	}
	
	
	
	
}




