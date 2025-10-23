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
                        if (x[12] <= 2.101499915122986) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[34] <= 0.2685000002384186) {
                                if (x[71] <= 6.33299994468689) {
                                    if (x[0] <= 1.0020000338554382) {
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

                            else {
                                if (x[12] <= 2.380500078201294) {
                                    votes[1] += 1;
                                }

                                else {
                                    votes[2] += 1;
                                }
                            }
                        }

                        // tree #2
                        if (x[71] <= 2.541000008583069) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[16] <= 0.2435000017285347) {
                                if (x[69] <= 3.27400004863739) {
                                    votes[2] += 1;
                                }

                                else {
                                    votes[1] += 1;
                                }
                            }

                            else {
                                if (x[43] <= 0.017500000074505806) {
                                    if (x[19] <= -0.7345000207424164) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[24] <= -0.11900000274181366) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[64] <= -0.0364999994635582) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }
                        }

                        // tree #3
                        if (x[13] <= -0.5839999914169312) {
                            if (x[31] <= 1.550499975681305) {
                                if (x[34] <= 0.15599999576807022) {
                                    votes[1] += 1;
                                }

                                else {
                                    votes[0] += 1;
                                }
                            }

                            else {
                                if (x[10] <= 2.2144999504089355) {
                                    if (x[18] <= 0.24899999052286148) {
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
                            if (x[38] <= 0.015500000212341547) {
                                votes[2] += 1;
                            }

                            else {
                                votes[2] += 1;
                            }
                        }

                        // tree #4
                        if (x[29] <= -0.012500000186264515) {
                            if (x[12] <= 2.44350004196167) {
                                if (x[71] <= 2.7829999923706055) {
                                    votes[0] += 1;
                                }

                                else {
                                    if (x[70] <= 0.7314999997615814) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[16] <= 0.22699999809265137) {
                                    if (x[71] <= 6.350499868392944) {
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
                            if (x[19] <= -0.5034999847412109) {
                                if (x[14] <= -0.5789999961853027) {
                                    if (x[19] <= -0.7510000169277191) {
                                        if (x[27] <= -0.019500000402331352) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            if (x[11] <= 1.312999963760376) {
                                                votes[2] += 1;
                                            }

                                            else {
                                                votes[1] += 1;
                                            }
                                        }
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
                                votes[0] += 1;
                            }
                        }

                        // tree #5
                        if (x[12] <= 2.1095000505447388) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[35] <= 0.7644999921321869) {
                                if (x[69] <= 3.615999937057495) {
                                    votes[2] += 1;
                                }

                                else {
                                    votes[1] += 1;
                                }
                            }

                            else {
                                if (x[13] <= -0.5774999856948853) {
                                    if (x[66] <= -0.03749999962747097) {
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
                        }

                        // tree #6
                        if (x[10] <= 2.5740000009536743) {
                            if (x[15] <= -0.5825000107288361) {
                                if (x[2] <= 0.4880000054836273) {
                                    votes[1] += 1;
                                }

                                else {
                                    if (x[69] <= 3.8585000038146973) {
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
                        if (x[15] <= -0.8534999787807465) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[35] <= 0.7644999921321869) {
                                if (x[2] <= 0.494499996304512) {
                                    votes[1] += 1;
                                }

                                else {
                                    if (x[35] <= 0.7339999973773956) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[38] <= 0.018499999307096004) {
                                    if (x[70] <= 0.5300000011920929) {
                                        if (x[70] <= 0.41499999165534973) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            if (x[17] <= 0.22950000315904617) {
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
                                    votes[2] += 1;
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