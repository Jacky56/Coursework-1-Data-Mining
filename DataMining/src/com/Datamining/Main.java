package com.Datamining;

import java.util.*;
import org.jblas.*;
import java.util.concurrent.*;

import javax.xml.crypto.Data;

public class Main {
	public static void main(String[] args) {
		
		DataManager.SetRegex("?");
		DataManager.SetSeed(10l); //debugging for deterministic random
		
		ArrayList<String[]> dataSet = DataManager.ReadCSV("data/adult.train.5fold.csv",false);
		dataSet = DataManager.ReplaceMissingValues(dataSet);
		DoubleMatrix M = DataManager.dataSetToMatrix(dataSet,14);
		List<DoubleMatrix> bins = DataManager.split(M, 15);
		//List<DoubleMatrix> bins = DataManager.splitFold(M, 5,true); //random via permutation debugging
		
		ArrayList<String[]> testSet = DataManager.ReadCSV("data/adult.test.csv",false);
		testSet = DataManager.ReplaceMissingValues(testSet);
		DoubleMatrix test = DataManager.dataSetToMatrix(testSet,14);
		List<DoubleMatrix> testBins = DataManager.splitFold(test, 8, false);
		
		
		//initiate threads 
		int threadPool = 16;
		ExecutorService executor = Executors.newFixedThreadPool(threadPool);
		List<Callable<Record[]>> callable = new ArrayList<Callable<Record[]>>();		
		
		long startTime = System.currentTimeMillis(); //debugging - test for optimisation 
		
		
		
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
			
			//build worker thread
			CrossValidation fold = new CrossValidation(Train, Validation,14,new int[]{14,15},40,2,false);
			callable.add(fold);
		}
		
		//returned statistics of each fold.
		List<Record[]> unweightedRecords = new ArrayList<Record[]>();
		
		try {
			//collect all work thread values 
			List<Future<Record[]>> set = executor.invokeAll(callable);
			
			for(Future<Record[]> recordFold : set) {
				unweightedRecords.add(recordFold.get());
			}
			
		} catch (Exception e) {
			// TODO: handle exception
		}
		
		
	
		callable.clear();
		
		
		
		//weighted
		System.out.println("Validating Weighted KNN...");
		for(int i = 0; i < bins.size(); i++) {
			DoubleMatrix Train = new DoubleMatrix(0,bins.get(0).columns);
			DoubleMatrix Validation = bins.get(i);
			
			for(int j = 0; j < bins.size(); j++) {
				if(i != j) {
					Train = DoubleMatrix.concatVertically(Train, bins.get(j));
				}
			}
			//build worker thread
			CrossValidation fold = new CrossValidation(Train, Validation,14,new int[]{14,15},40,2,true);
			callable.add(fold);
		}
		
		//returned statistics of each fold.
		List<Record[]> weightedRecords = new ArrayList<Record[]>();
		
		try {
			//collect all work thread values 
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
		int validationIndex = 0;
		for(int i = 0; i < unweightedRecords.get(0).length; i++) {
			
			int currentK = 0;
			double currentAccuracy = 0;
			
			List<Record[]> a = new ArrayList<Record[]>();
			for(int j = 0; j < unweightedRecords.size(); j ++) { 
				a.add(new Record[]{ unweightedRecords.get(j)[i]});
				currentK = unweightedRecords.get(j)[i].KNN;
			}
			
			int[][] m = DataManager.combineMat(a);
			currentAccuracy = DataManager.getAccuracy(m);
			
			if(currentAccuracy > bestAccuracy) {
				bestKNN = currentK;
				bestAccuracy = currentAccuracy;
				weighted = false;
				validationIndex = i;
				System.out.println(bestKNN + " : unweighted : " +bestAccuracy  + " : Best");
			} else {
				System.out.println(currentK + " : unweighted : " + currentAccuracy);
			}
		}
		
		
		for(int i = 0; i < weightedRecords.get(0).length; i++) {
			
			int currentK = 0;
			double currentAccuracy = 0;
			
			List<Record[]> a = new ArrayList<Record[]>();
			for(int j = 0; j < weightedRecords.size(); j ++) { 
				a.add(new Record[]{ weightedRecords.get(j)[i]});
				currentK = weightedRecords.get(j)[i].KNN;
			}
			
			int[][] m = DataManager.combineMat(a);
			currentAccuracy = DataManager.getAccuracy(m);
			
			if(currentAccuracy > bestAccuracy) {
				bestKNN = currentK;
				bestAccuracy = currentAccuracy;
				weighted = true;
				validationIndex = i;
				System.out.println(bestKNN + " : weighted :" +bestAccuracy  + " : Best");
			} else {
				System.out.println(currentK + " : weighted :" + currentAccuracy);
			}
		}		
		
		
		List<Record[]> bestValidation = new ArrayList<Record[]>();
		for(int i = 0; i < bins.size(); i++) {
			if(weighted) {
				bestValidation.add(new Record[]{ weightedRecords.get(i)[validationIndex]});
			} else {
				bestValidation.add(new Record[]{ unweightedRecords.get(i)[validationIndex]});
			}

		}
		
		
		
		
		System.out.println("Testing...");
		
		//KNN on test set
		callable.clear();
		
		//build worker threads
		for(int i = 0; i < testBins.size(); i++) {
			CrossValidation fold = new CrossValidation(M, testBins.get(i),14,new int[]{14, 15},bestKNN,bestKNN,weighted);
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
		
		
		
		
		
		int[][] validationM =  DataManager.combineMat(bestValidation);
		int[][] testM = DataManager.combineMat(testRecords);
		double testAccuracy = DataManager.getAccuracy(testM);
		
		
		//print stuff
		System.out.println(bestKNN + " : "+ weighted + " : " + testAccuracy);
		System.out.println(testM[0][0] + ", " + testM[0][1] +", " + testM[1][0] + ", " + testM[1][1]);
		
		//delete all worker threads
		executor.shutdownNow();
		
		
		long endTime   = System.currentTimeMillis();
		long totalTime = endTime - startTime;
		System.out.println("Run time(millisecond): " + totalTime );
		System.out.println("Thread pool: " + threadPool );
		System.out.println("Cores: " + Runtime.getRuntime().availableProcessors());
		
		//prints to file
		DataManager.saveRecord("data/grid.results.txt", bestKNN, weighted,validationM, testM, testRecords.get(0)[0].classType, 5, totalTime, threadPool);
	}
	
	
	
	
}




