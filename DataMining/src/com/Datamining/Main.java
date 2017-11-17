package com.Datamining;

import java.util.*;
import org.jblas.*;
import java.util.concurrent.*;

public class Main {
	public static void main(String[] args) {
		
		ArrayList<String[]> dataSet = DataManager.ReadCSV("data/adult.train.5fold.csv",false);
		
		DataManager.SetRegex("?");
		
		dataSet = DataManager.ReplaceMissingValues(dataSet);
		
		DoubleMatrix M = DataManager.dataSetToMatrix(dataSet);
		
		List<DoubleMatrix> bins = DataManager.split(M, 15);
		
		
		
		
		//DoubleMatrix Xn = DataManager.Normalize(DataManager.GetFeatures(M, new int[]{14,15}));
		
		//System.out.println(DataManager.GetClass(M,14));
		
		//System.out.println(Xn);
		
		
		//--- debug: test if ? get replaced by common
//		System.out.println(dataSet.get(18)[13]);
//		dataSet = DataManager.RefactorDataSet(dataSet,"?");
//		System.out.println(dataSet.get(18)[13]);
		
		
		long startTime = System.currentTimeMillis();
		
		
		
		ExecutorService executor = Executors.newFixedThreadPool(16);
		List<Callable<Record>> callable = new ArrayList<Callable<Record>>();
		
		
		for(int i = 0; i < bins.size(); i++) {
			
			DoubleMatrix Train = new DoubleMatrix(0,bins.get(0).columns);
			
			
			for(int j = 0; j < bins.size(); j++) {
				if(i != j) {
					Train = DoubleMatrix.concatVertically(Train, bins.get(j));
				}
			}
			
			CrossValidation fold = new CrossValidation(Train, bins.get(i),14,new int[]{14,15},40,2);
			callable.add(fold);
			System.out.println("Work threads: " + callable.size());
		}
		
		try {
			List<Future<Record>> set = executor.invokeAll(callable);
			for(Future<Record> a : set) {
				
				System.out.println(a.get().accuracy);
			}
		} catch (Exception e) {
			// TODO: handle exception
		}
		
		executor.shutdownNow();
		
		
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Run time(millisecond): " + totalTime);
		
		
	}
	
	
	
	
}




