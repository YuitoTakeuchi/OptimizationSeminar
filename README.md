# 最適化ゼミ

C++で実装していく．

# 使い方
CMakeでビルドできます．
```zsh
$ cd OptimizationSeminar
$ mkdir build
$ cd build
$ cmake ..
$ make
```
手元環境の関係でminimum_requiredを3.25にしていて，もしかしたら古いCMakeだとビルドできないかもしれません．その時はCMakeをアップデートするかCMakeLists.txtを書き換えてください．
```zsh
$ cmake --version
cmake version 3.26.4

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

## Chapter 4. Unconstrained Gradient-Based Optimization

### 直線探索ベースの手法
信頼領域法以外の手法が該当します．直線探索手法としてはArmijo条件を使うもの，Wolf条件を使うものを実装してあります．（Wolfは今のところバグっていてステップ幅が小さいと抜けられなくなります）  
探索方向には再急降下法，共役勾配法，ニュートン法，BFGSが実装してあります．  
共役勾配法はFletcher–Reeves formulaで実装しています．  


1. SearchDirectionをincludeする．
```cpp
#include "SearchDirection.hpp"
```

1. 目的関数，勾配関数，ニュートン法の場合はヘッシアン関数を記述する．
```cpp
double func(const Eigen::VectorXd& point) {
    double ret = 0.0;
    double x1 = point(0), x2 = point(1);
    ret += (1.0-x1)*(1.0-x1) + (1.0-x2)*(1.0-x2) + 0.5*(2.0*x2-x1*x1)*(2.0*x2-x1*x1);
    return ret;
}

Eigen::VectorXd calc_grad(const Eigen::VectorXd& point) {
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(point.rows());
    double x1 = point(0), x2 = point(1);
    ret(0) = -2+2.0*x1-4.0*x1*x2+2.0*x1*x1*x1;
    ret(1) = -2.0+6.0*x2-2.0*x1*x1;
    return ret;
}

// Hessian is only needed for Newton's method
Eigen::MatrixXd calc_hessian(const Eigen::VectorXd& x) {
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(2, 2);
    double x1 = x(0), x2 = x(1);
    ret(0, 0) = 2.0-4.0*x2+6.0*x1*x1;
    ret(0, 1) = -4.0*x1;
    ret(1, 0) = -4.0*x1;
    ret(1, 1) = 6.0;
    return ret;
}
```

3. LineSearchのアルゴリズムを指定し，最適化問題を解くインスタンスを作成する．メソッドを呼んで解く．
```cpp
    ConjugateGradient<Armijo> cg(&func, &calc_grad); // 直線探索にはArmijo条件を使用する．
    Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
    x << -1, 1; // 初期点
    cg.solve(x, 1e-6); // 初期点をx，toleranceを1e-6として問題を解く．
    cg.output_to_file("result.txt") // reuslt.txtに過程を出力．
```
結果の出力フォーマットは
```
x1 x2 x3 ... xn obective_function_value
...
```
です．


## Reference 
Martins, J. R. R. A. and Ning, A., [Engineering Design Optimization](https://mdobook.github.io/), Cambridge University Press, 2022.
