#!/usr/bin/env wolframscript

(*Case 1: initial r, v, T provided for periodic orbit*)
(*command line args:
Int 0
String "SE" or "EM" for Sun Earth or Earth Moon system
7 numbers representing r,v,T in AU and radians canonical time unit, or in km and seconds
String filename base
Int number of refinements (2^n) samplings at max refinement*)

(*Case 2: find a halo orbit*)
(*command line args: 
Int 1
String "SE" or "EM" for Sun Earth or Earth Moon system
String "N" or "S" for northern or southern family
Float Az (out of plane amplitude of the halo)
String filename base
Int number of refinements (2^n) samplings at max refinement*)



(*command line inputs*)
argv = Rest @ $ScriptCommandLine;
argv = argv
argc = Length @ argv;


(*perform input validation and parse data*)
If[(argc != 6 && argc != 11), Print["Incorrect number of arguments to wolfram script, exiting."]; Exit[]]

(*common inputs*)
SunEarthFlag = (argv[[2]]=="SE");
filename = argv[[argc-1]];
numRefinements = ToExpression[argv[[argc]]];

case = ToExpression[argv[[1]]];

(*Case 1  or 2 inputs*)
If[case == 0, 
r0 = ToExpression[argv[[3;;5]]]; v0 = ToExpression[argv[[6;;8]]]; T = ToExpression[argv[[9]]], 
If[argv[[3]] == "N" , NSFlag = 1, NSFlag = 3]; Az = ToExpression[argv[[4]]]];



(*Define necessary functions and constants*)

Needs["DifferentialEquations`InterpolatingFunctionAnatomy`"];
session = StartExternalSession["Python"];
ExternalEvaluate[session, 
  File["./cr3bp.py"]];

(*change to Earth Moon constants if requested*)
If[(!SunEarthFlag), ExternalEvaluate[session, "setMoon" -> {}]];

mu1 = ExternalEvaluate[session, "getMu" -> {}];
L2 = ExternalEvaluate[session, "getL2" -> {}];
U = 1/2 (x^2 + y^2) + (1 - mu)/r1 + 
     mu/r2 /. {r1 -> Sqrt[(mu + x)^2 + y^2 + z^2], 
     r2 -> Sqrt[(1 - mu - x)^2 + y^2 + z^2]} /. {x -> x[t], y -> y[t],
     z -> z[t]};
eqns = {2 y'[t] + D[U, x[t]], -2 x'[t] + D[U, y[t]], D[U, z[t]]};
eqnsmu = eqns /. {mu -> mu1};
jacmu = D[
   Join[{x'[t], y'[t], z'[t]}, 
    eqnsmu], {{x[t], y[t], z[t], x'[t], y'[t], z'[t]}}];
findHaloIterate[ics_] := 
 Module[{soln, pos, vel, dif, next, stm, T}, 
  soln = NDSolve[{eqnsmu == {x''[t], y''[t], z''[t]}, 
     Phi'[t] == 
      jacmu . Phi[t], {x[0], y[0], z[0], x'[0], y'[0], z'[0]} == ics, 
     Phi[0] == IdentityMatrix[6]}, {x, y, z, Phi}, {t, Infinity}, 
    Method -> {"EventLocator", "Event" -> y[t], 
      "EventLocationMethod" -> "LinearInterpolation"},
	AccuracyGoal -> 10, PrecisionGoal -> 10];
  Sow[T = InterpolatingFunctionDomain[First[y /. soln]][[1, -1]]];
  pos = ({x[T], y[T], z[T]} /. soln)[[1]];
  vel = ({x'[T], y'[T], z'[T]} /. soln)[[1]];
  stm = ArrayReshape[(Phi /. soln)[[1]][T], {6, 6}];
  (*d(xd_f, zd_f)/d(x_0,yd_0)*)
  dif = stm[[{4, 6}, {1, 5}]];
  next = ics - {#[[1]], 0, 0, 0, #[[2]], 0} &[
    LinearSolve[dif, {vel[[1]], vel[[3]]}]]]
findHalo[ics_] := {#[[1]], 2*#[[2, 1]][[-1]]} &[
  Reap[FixedPoint[findHaloIterate[#] &, ics, 1000, 
    SameTest -> (Norm[#1 - #2] < 10^(-9) &)]]]
findHaloAz[Az_] := 
 findHalo[Join[Join[#[[1 ;; 2]], {Sign[#[[3]]] Az}], #[[4 ;; 6]]] &[
   ExternalEvaluate[session, 
     "getThirdOrderHaloApprox" -> {Az, NSFlag, 0}][[;; -2]]]]

PlotCR3BP[ics_, T_] := 
 Module[{soln}, 
  soln = NDSolve[{eqnsmu == {x''[t], y''[t], z''[t]}, 
     Phi'[t] == 
      jacmu . Phi[t], {x[0], y[0], z[0], x'[0], y'[0], z'[0]} == ics, 
     Phi[0] == IdentityMatrix[6]}, {x, y, z, Phi}, {t, 0, T}];
  {ParametricPlot[{x[t], y[t]} /. soln[[1]], {t, 0, T}], 
   ParametricPlot3D[{x[t], y[t], z[t]} /. soln[[1]], {t, 0, T}], 
   Plot[Det[(Phi[t] /. soln[[1]])[[1 ;; 3, 4 ;; 6]]], {t, 0, T}]}]



(*propagate some time, get final position and stm*)
propSTM[ics_, dt_?NumberQ] := Module[{soln},
  soln = NDSolve[{eqnsmu == {x''[t], y''[t], z''[t]}, 
     Phi'[t] == 
      jacmu . Phi[t], {x[0], y[0], z[0], x'[0], y'[0], z'[0]} == ics, 
     Phi[0] == IdentityMatrix[6]}, {x, y, z, Phi}, {t, dt, dt}, 
     AccuracyGoal -> 14, PrecisionGoal -> 14];
  {({x[dt], y[dt], z[dt], x'[dt], y'[dt], z'[dt]} /. 
     soln[[1]]), (Phi[dt] /. soln[[1]])[[1 ;; 6, 1 ;; 6]]}
  ]


(*END definition of functions and constants*)



(*find orbit if case 2 or package r0 and v0 into x0*)
If[case == 1, res = findHaloAz[Az]; x0 = res[[1]]; T = res[[2]], x0 = Join[r0, v0]]


(*plot and export picture of the reference orbit*)
images = PlotCR3BP[x0, T]
Export["./" <> filename <> "_orbit2d.png", images[[1]]]
Export["./" <> filename <> "_orbit3d.png", images[[2]]]

(*compute and store orbit and STM*)
dt = T/(2^numRefinements);


data = NestList[propSTM[#[[1]], dt] &, {x0, IdentityMatrix[6]}, 2^numRefinements];
dataX = data[[All, 1]];
dataPhi = data[[All, 2]];
dataXandT = Table[Join[{i*dt},dataX[[i+1]]], {i,0,2^numRefinements}];


(*export t, r, v into one file*)
Export["./" <> filename <> "_trvs.mat", dataXandT];

(*export Phi_{t_i,t_i+1} to another file*)
Export["./" <> filename <> "_STMs.mat", dataPhi];
