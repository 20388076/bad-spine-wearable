#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= 1.003499984741211) {
                            if (x[1] <= 0.019499999471008778) {
                                return 14;
                            }

                            else {
                                if (x[1] <= 0.02650000061839819) {
                                    if (x[2] <= 0.08349999785423279) {
                                        if (x[2] <= 0.06350000202655792) {
                                            if (x[0] <= 0.9814999997615814) {
                                                return 3;
                                            }

                                            else {
                                                return 12;
                                            }
                                        }

                                        else {
                                            if (x[1] <= 0.02350000012665987) {
                                                return 12;
                                            }

                                            else {
                                                return 0;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[0] <= 0.9805000126361847) {
                                            if (x[1] <= 0.024500000290572643) {
                                                return 9;
                                            }

                                            else {
                                                return 6;
                                            }
                                        }

                                        else {
                                            if (x[3] <= 0.013500000350177288) {
                                                return 3;
                                            }

                                            else {
                                                return 9;
                                            }
                                        }
                                    }
                                }

                                else {
                                    if (x[2] <= 0.11549999937415123) {
                                        if (x[2] <= 0.10350000113248825) {
                                            if (x[0] <= 0.9695000052452087) {
                                                return 13;
                                            }

                                            else {
                                                return 3;
                                            }
                                        }

                                        else {
                                            if (x[1] <= 0.04350000061094761) {
                                                return 1;
                                            }

                                            else {
                                                return 14;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[3] <= 0.014500000048428774) {
                                            if (x[0] <= 0.9705000221729279) {
                                                return 13;
                                            }

                                            else {
                                                return 10;
                                            }
                                        }

                                        else {
                                            if (x[2] <= 0.2524999976158142) {
                                                return 4;
                                            }

                                            else {
                                                return 7;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        else {
                            if (x[1] <= 0.035499999299645424) {
                                if (x[3] <= 0.012500000186264515) {
                                    if (x[0] <= 1.0235000252723694) {
                                        if (x[2] <= 0.1574999988079071) {
                                            if (x[8] <= 0.0005000000237487257) {
                                                return 5;
                                            }

                                            else {
                                                return 5;
                                            }
                                        }

                                        else {
                                            return 8;
                                        }
                                    }

                                    else {
                                        if (x[0] <= 1.0444999933242798) {
                                            if (x[2] <= 0.21299999952316284) {
                                                return 11;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }

                                        else {
                                            if (x[2] <= 0.21700000017881393) {
                                                return 8;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }
                                    }
                                }

                                else {
                                    if (x[1] <= 0.02650000061839819) {
                                        if (x[2] <= 0.13849999755620956) {
                                            if (x[0] <= 1.0205000042915344) {
                                                return 8;
                                            }

                                            else {
                                                return 8;
                                            }
                                        }

                                        else {
                                            if (x[0] <= 1.0335000157356262) {
                                                return 11;
                                            }

                                            else {
                                                return 8;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[1] <= 0.027500000782310963) {
                                            if (x[0] <= 1.018500030040741) {
                                                return 11;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }

                                        else {
                                            if (x[0] <= 1.0105000138282776) {
                                                return 5;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[1] <= 0.038999998942017555) {
                                    if (x[0] <= 1.027999997138977) {
                                        return 2;
                                    }

                                    else {
                                        return 8;
                                    }
                                }

                                else {
                                    if (x[8] <= 0.0025000000605359674) {
                                        if (x[1] <= 0.23799999803304672) {
                                            if (x[0] <= 1.0164999961853027) {
                                                return 5;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }

                                        else {
                                            if (x[2] <= 0.10649999976158142) {
                                                return 5;
                                            }

                                            else {
                                                return 14;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[2] <= 0.19350000470876694) {
                                            if (x[0] <= 1.0224999785423279) {
                                                return 5;
                                            }

                                            else {
                                                return 5;
                                            }
                                        }

                                        else {
                                            return 11;
                                        }
                                    }
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }