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
                        if (x[12] <= -0.65430848300457) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[34] <= 0.8241652846336365) {
                                if (x[74] <= -0.29880863428115845) {
                                    if (x[71] <= 0.8632164299488068) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[19] <= -0.7547949254512787) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[33] <= -0.5668039321899414) {
                                            votes[2] += 1;
                                        }

                                        else {
                                            votes[1] += 1;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[69] <= -0.40035511553287506) {
                                    votes[2] += 1;
                                }

                                else {
                                    votes[1] += 1;
                                }
                            }
                        }

                        // tree #2
                        if (x[71] <= -0.7634138762950897) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[16] <= 0.37342871725559235) {
                                if (x[38] <= -0.3988892436027527) {
                                    if (x[0] <= -0.922590434551239) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }

                                else {
                                    if (x[69] <= -0.6799655258655548) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        votes[1] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[37] <= -0.09051985666155815) {
                                    if (x[12] <= 0.38420137763023376) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[40] <= 1.0529203116893768) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }
                        }

                        // tree #3
                        if (x[13] <= 0.5364863574504852) {
                            if (x[31] <= 0.21854743361473083) {
                                if (x[34] <= -1.23800927400589) {
                                    votes[1] += 1;
                                }

                                else {
                                    votes[0] += 1;
                                }
                            }

                            else {
                                if (x[10] <= -0.5326173007488251) {
                                    if (x[17] <= 0.43130141496658325) {
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
                            if (x[13] <= 0.5484513640403748) {
                                votes[2] += 1;
                            }

                            else {
                                votes[2] += 1;
                            }
                        }

                        // tree #4
                        if (x[59] <= -0.3782835304737091) {
                            if (x[12] <= 0.5365630984306335) {
                                if (x[71] <= -0.4967955816537142) {
                                    votes[0] += 1;
                                }

                                else {
                                    if (x[70] <= 0.9174395799636841) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }
                            }

                            else {
                                if (x[16] <= 0.10660528391599655) {
                                    if (x[71] <= 0.911546528339386) {
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
                            if (x[18] <= -0.07965419068932533) {
                                if (x[14] <= 0.5102002769708633) {
                                    if (x[20] <= 0.6376804560422897) {
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
                                if (x[51] <= 0.5892100632190704) {
                                    if (x[69] <= 0.5456845350563526) {
                                        if (x[28] <= 0.019134512171149254) {
                                            if (x[23] <= -0.3688352406024933) {
                                                if (x[33] <= -0.5150444805622101) {
                                                    votes[2] += 1;
                                                }

                                                else {
                                                    if (x[39] <= -0.12855172529816628) {
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
                                            if (x[14] <= 0.5781081020832062) {
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
                                    if (x[11] <= -1.6179879903793335) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        if (x[19] <= -0.459921732544899) {
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
                        if (x[12] <= -0.6262879222631454) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[10] <= -0.6337232291698456) {
                                votes[2] += 1;
                            }

                            else {
                                if (x[69] <= -0.5477139949798584) {
                                    if (x[54] <= -0.21992338448762894) {
                                        if (x[71] <= 0.5693607032299042) {
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
                        if (x[10] <= 0.7872730195522308) {
                            if (x[15] <= 0.5813391208648682) {
                                if (x[2] <= 1.1526972651481628) {
                                    if (x[69] <= -0.6592563390731812) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[14] <= 0.553338497877121) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            votes[2] += 1;
                                        }
                                    }
                                }

                                else {
                                    if (x[44] <= -0.5950546562671661) {
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
                        if (x[15] <= -0.8962883651256561) {
                            votes[0] += 1;
                        }

                        else {
                            if (x[35] <= 0.6173312366008759) {
                                if (x[2] <= 1.0096973478794098) {
                                    votes[1] += 1;
                                }

                                else {
                                    if (x[17] <= 0.7270714342594147) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        if (x[65] <= -0.6441311687231064) {
                                            votes[1] += 1;
                                        }

                                        else {
                                            if (x[20] <= -0.4881940633058548) {
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
                                if (x[53] <= -0.5387172996997833) {
                                    if (x[58] <= -0.5210682153701782) {
                                        votes[1] += 1;
                                    }

                                    else {
                                        votes[2] += 1;
                                    }
                                }

                                else {
                                    if (x[69] <= -0.639597088098526) {
                                        votes[2] += 1;
                                    }

                                    else {
                                        if (x[14] <= 0.553338497877121) {
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