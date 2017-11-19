package com.Datamining;

import java.util.*;
import org.jblas.*;
import java.util.concurrent.*;

import javax.xml.crypto.Data;

public class Main {
	public static void main(String[] args) {
		
		DataManager.SetRegex("?");

		ArrayList<String[]> dataSet = DataManager.ReadCSV("data/adult.train.5fold.csv",false);
		dataSet = DataManager.ReplaceMissingValues(dataSet);
		DoubleMatrix M = DataManager.dataSetToMatrix(dataSet,14);
		List<DoubleMatrix> bins = DataManager.split(M, 15);

		ArrayList<String[]> testSet = DataManager.ReadCSV("data/adult.test.csv",false);
		testSet = DataManager.ReplaceMissingValues(testSet);
		DoubleMatrix test = DataManager.dataSetToMatrix(testSet,14);
		List<DoubleMatrix> testBins = DataManager.splitFold(test, 8);
		
		//DoubleMatrix Xn = DataManager.Normalize(DataManager.GetFeatures(M, new int[]{14,15}));
		
		//System.out.println(DataManager.GetClass(M,14));
		
		//System.out.println(Xn);
		
		
		//--- debug: test if ? get replaced by common
//		System.out.println(dataSet.get(18)[13]);
//		dataSet = DataManager.RefactorDataSet(dataSet,"?");
//		System.out.println(dataSet.get(18)[13]);
		

		int threadPool = 16;
		ExecutorService executor = Executors.newFixedThreadPool(threadPool);
		List<Callable<Record[]>> callable = new ArrayList<Callable<Record[]>>();		
		
		long startTime = System.currentTimeMillis();
		
		
		//unweighted
		System.out.println("Validating Unweighted KNN...");
		for(int i = 0; i < bins.size(); i++) {
			DoubleMatrix Train = new DoubleMatrix(0,bins.get(0).columns);
			DoubleMatrix Validation = bins.get(i);
			
			for(int j = 0; j < bins.size(); j++) {
				if(i != j) {
					Train = DoubleMatrix.concatVertically(Train, bins.get(j));
				}
			}
			
			CrossValidation fold = new CrossValidation(Train, Validation,14,new int[]{14,15},40,2,false);
			callable.add(fold);
			//System.out.println("Work threads: " + callable.size());
		}
		
		//---returned statistics of each fold.
		List<Record[]> unweightedRecords = new ArrayList<Record[]>();
		
		try {
			List<Future<Record[]>> set = executor.invokeAll(callable);
			
			for(Future<Record[]> recordFold : set) {
				unweightedRecords.add(recordFold.get());
			}
			
		} catch (Exception e) {
			// TODO: handle exception
		}
		
		
	
		
		
		
		
		
		//weighted
		callable.clear();
		System.out.println("Validating Weighted KNN...");
		for(int i = 0; i < bins.size(); i++) {
			DoubleMatrix Train = new DoubleMatrix(0,bins.get(0).columns);
			DoubleMatrix Validation = bins.get(i);
			
			for(int j = 0; j < bins.size(); j++) {
				if(i != j) {
					Train = DoubleMatrix.concatVertically(Train, bins.get(j));
				}
			}
			
			CrossValidation fold = new CrossValidation(Train, Validation,14,new int[]{14,15},40,2,true);
			callable.add(fold);
			//System.out.println("Work threads: " + callable.size());
		}
		
		//---returned statistics of each fold.
		List<Record[]> weightedRecords = new ArrayList<Record[]>();
		
		try {
			List<Future<Record[]>> set = executor.invokeAll(callable);
			
			for(Future<Record[]> recordFold : set) {
				weightedRecords.add(recordFold.get());
			}
			
		} catch (Exception e) {
			// TODO: handle exception
		}		
		
		
		
		
		
		
		
		
		//find best parameter 
		int bestKNN = 0;
		double bestAccuracy = 0;
		boolean weighted = false;
		
		
		for(int i = 0; i < unweightedRecords.get(0).length; i++) {
			
			int currentK = 0;
			double currentAccuracy = 0;
			for(int j = 0; j < unweightedRecords.size(); j ++) {
				currentAccuracy += unweightedRecords.get(j)[i].accuracy;
				currentK = unweightedRecords.get(j)[i].KNN;
			}
			
			if(currentAccuracy > bestAccuracy) {
				bestKNN = currentK;
				bestAccuracy = currentAccuracy;
				weighted = false;
				System.out.println(bestKNN + " : unweighted : " +bestAccuracy/unweightedRecords.size()  + " : Best");
			} else {
				System.out.println(currentK + " : unweighted : " + currentAccuracy/unweightedRecords.size());
			}
		}
		
		for(int i = 0; i < weightedRecords.get(0).length; i++) {
			
			int currentK = 0;
			double currentAccuracy = 0;
			for(int j = 0; j < weightedRecords.size(); j ++) {
				currentAccuracy += weightedRecords.get(j)[i].accuracy;
				currentK = weightedRecords.get(j)[i].KNN;
			}
			
			if(currentAccuracy > bestAccuracy) {
				bestKNN = currentK;
				bestAccuracy = currentAccuracy;
				weighted = true;
				System.out.println(bestKNN + " : weighted :" +bestAccuracy/weightedRecords.size()  + " : Best");
			} else {
				System.out.println(currentK + " : weighted :" + currentAccuracy/weightedRecords.size());
			}
		}		
		
		
		
		
		
		
		
		
		
		
		
		
		System.out.println("Testing...");
		
		//KNN on test set
		callable.clear();
		
		for(int i = 0; i < testBins.size(); i++) {
			CrossValidation fold = new CrossValidation(M, testBins.get(i),14,new int[]{14, 15},bestKNN,bestKNN-1,weighted);
			callable.add(fold);
		}
		List<Record[]> testRecords = new ArrayList<Record[]>();
		
		try {
			List<Future<Record[]>> set = executor.invokeAll(callable);
			
			for(Future<Record[]> recordFold : set) {
				testRecords.add(recordFold.get());
			}
		} catch (Exception e) {
			// TODO: handle exception
		}
		
		
		int[][] testM = new int[2][2];
		for(Record[] a : testRecords) {
			for(int i = 0; i < 2; i++) {
				for(int j = 0; j < 2; j++) {
					testM[i][j] += a[1].confusionMat[i][j];
				}
			}
		}
		
		int correct = 0;
		int total = 0;
		for(int i = 0; i < 2; i++) {
			for(int j = 0; j < 2; j++) {
				total += testM[i][j];
				if(i == j) {
					correct += testM[i][j];
				}
			}
		}
		
		double testAccuracy = (double)correct / total;
		
		System.out.println(bestKNN + " : "+ weighted + " : " + testAccuracy);
		System.out.println(testM[0][0] + ", " + testM[0][1] +", " + testM[1][0] + ", " + testM[1][1]);
		
		
		//prints to file
		DataManager.saveRecord("data/grid.results.txt", bestKNN, weighted, testM, testRecords.get(0)[0].classType, testAccuracy);
		
		executor.shutdownNow();
		
		
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Run time(millisecond): " + totalTime );
		System.out.println("Thread pool: " + threadPool );
		
//		for(Record a : records.get(0)) {
//			System.out.println(a.accuracy + " : " + a.KNN);
//		}
//		System.out.println("");
		
		
		//System.out.println(records.get(0)[4].KNN);
		
		
		
	}
	
	
	
	
}




