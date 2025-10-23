#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class RandomForest {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        uint8_t votes[3] = { 0 };
                        // tree #1
                        if (x[12] <= 2.10099995136261) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[34] <= 2.631500005722046) {
                                if (x[74] <= 0.0035000001080334187) {
                                    if (x[71] <= 634.572509765625) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[19] <= -7.791000127792358) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[33] <= 5.674499988555908) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[69] <= 382.99951171875) {
                                    votes[2] += 1;
                                }

                                else {
                                    votes[1] += 1;
                                }
                            }
                        }

                        // tree #2
                        if (x[71] <= 244.3574981689453) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[16] <= 2.3890000581741333) {
                                if (x[38] <= 0.018499999307096004) {
                                    if (x[0] <= 9.754000186920166) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[69] <= 315.0924987792969) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[37] <= 0.025500000454485416) {
                                    if (x[12] <= 2.397499918937683) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[40] <= 0.10399999842047691) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }
                        }

                        // tree #3
                        if (x[13] <= -5.728500127792358) {
                            if (x[31] <= 15.203500270843506) {
                                if (x[34] <= 1.527999997138977) {
                                    votes[1] += 1;
                                }

                                else {
                                    votes[0] += 1;
                                }
                            }

                            else {
                                if (x[10] <= 2.215000033378601) {
                                    if (x[17] <= 2.499500036239624) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    votes[1] += 1;
                                }
                            }
                        }

                        else {
                            if (x[13] <= -5.707000017166138) {
                                votes[2] += 1;
                            }

                            else {
                                votes[2] += 1;
                            }
                        }

                        // tree #4
                        if (x[59] <= 0.02350000012665987) {
                            if (x[12] <= 2.440999984741211) {
                                if (x[71] <= 308.31700134277344) {
                                    votes[0] += 1;
                                }

                                else {
                                    if (x[70] <= 69.64199829101562) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[16] <= 2.2454999685287476) {
                                    if (x[71] <= 646.16650390625) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    votes[2] += 1;
                                }
                            }
                        }

                        else {
                            if (x[18] <= 2.0605000257492065) {
                                if (x[14] <= -5.704999923706055) {
                                    if (x[20] <= -4.692499995231628) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[0] += 1;
                                    }
                                }

                                else {
                                    votes[2] += 1;
                                }
                            }

                            else {
                                if (x[51] <= 0.07250000163912773) {
                                    if (x[69] <= 612.7575073242188) {
                                        if (x[28] <= -0.017500000074505806) {
                                            if (x[23] <= -0.003999999957159162) {
                                                if (x[33] <= 5.767499923706055) {
                                                    votes[2] += 1;
                                                }

                                                else {
                                                    if (x[39] <= 0.02950000111013651) {
                                                        votes[1] += 1;
                                                    }

                                                    else {
                                                        votes[2] += 1;
                                                    }
                                                }
                                            }

                                            else {
                                                votes[2] += 1;
                                            }
                                        }

                                        else {
                                            if (x[14] <= -5.58299994468689) {
                                                votes[1] += 1;
                                            }

                                            else {
                                                votes[2] += 1;
                                            }
                                        }
                                    }

                                    else {
                                        votes[0] += 1;
                                    }
                                }

                                else {
                                    if (x[11] <= 1.2545000314712524) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        if (x[19] <= -7.16100001335144) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        // tree #5
                        if (x[12] <= 2.1089999675750732) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[10] <= 2.1875) {
                                votes[2] += 1;
                            }

                            else {
                                if (x[69] <= 347.2115020751953) {
                                    if (x[54] <= 0.011500000022351742) {
                                        if (x[71] <= 564.0789794921875) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    votes[1] += 1;
                                }
                            }
                        }

                        // tree #6
                        if (x[10] <= 2.5740000009536743) {
                            if (x[15] <= -5.7149999141693115) {
                                if (x[2] <= 4.871999979019165) {
                                    if (x[69] <= 320.1219940185547) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[14] <= -5.62749981880188) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[2] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[44] <= 0.0025000000605359674) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }
                            }

                            else {
                                votes[2] += 1;
                            }
                        }

                        else {
                            votes[0] += 1;
                        }

                        // tree #7
                        if (x[15] <= -8.371500015258789) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[35] <= 7.496999979019165) {
                                if (x[2] <= 4.797999858856201) {
                                    votes[1] += 1;
                                }

                                else {
                                    if (x[17] <= 2.6565001010894775) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        if (x[65] <= -0.20100000500679016) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            if (x[20] <= -7.14549994468689) {
                                                votes[2] += 1;
                                            }

                                            else {
                                                votes[1] += 1;
                                            }
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[53] <= 0.009499999694526196) {
                                    if (x[58] <= 0.014500000048428774) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[69] <= 324.89649963378906) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[14] <= -5.62749981880188) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[2] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        // return argmax of votes
                        uint8_t classIdx = 0;
                        float maxVotes = votes[0];

                        for (uint8_t i = 1; i < 3; i++) {
                            if (votes[i] > maxVotes) {
                                classIdx = i;
                                maxVotes = votes[i];
                            }
                        }

                        return classIdx;
                    }

                protected:
                };
            }
        }
    }