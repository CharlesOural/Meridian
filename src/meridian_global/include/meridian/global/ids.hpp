#pragma once

#include "meridian/core/strong_id.hpp"

namespace meridian::global {

struct GlobalGraphRevisionTag;
struct GlobalFactorIdTag;
struct ProposalIdTag;
struct CandidateIdTag;

using GlobalGraphRevision = core::StrongId<GlobalGraphRevisionTag>;
using GlobalFactorId = core::StrongId<GlobalFactorIdTag>;
using ProposalId = core::StrongId<ProposalIdTag>;
using CandidateId = core::StrongId<CandidateIdTag>;

}  // namespace meridian::global
