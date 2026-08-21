export const DURATION_OPTIONS = [
  [3600, '1 hour'],
  [21600, '6 hours'],
  [86400, '24 hours'],
  [259200, '3 days'],
  [604800, '7 days'],
  [2592000, '30 days'],
]

export const SESSION_OPTIONS = [1, 2, 3, 4, 5]

export const DEFAULT_DURATION_SECONDS = String(DURATION_OPTIONS[0][0])

export const INVITATION_PAGE_SIZE = 8

export const LOG_PAGE_SIZE = 12

export const ADMIN_HINTS = {
  accessRecords: 'Total number of access records currently returned by Administration, including active, pending, expired and restricted records.',
  pending: 'Invitations that were generated but have not yet completed the required Google identity verification.',
  active: 'Access records that have been claimed successfully or are currently active for the verified Google identity.',
  restricted: 'Records that are expired, revoked, blocked or still use the legacy unverified access model.',
  guestName: 'Friendly name used by the Administrator to identify the person receiving this access invitation.',
  authorizedEmail: 'Google account that is authorized to claim the generated invitation. A different Google identity cannot claim it.',
  role: 'Permission profile assigned to the invitation.',
  duration: 'How long the generated access authorization remains valid before it expires.',
  sessions: 'Maximum number of simultaneously active authenticated sessions allowed for this access record.',
  status: 'Current lifecycle state of the invitation or identity-bound access record.',
  claimedIdentity: 'Google identity that successfully claimed the invitation. It remains bound to the access record after verification.',
  expires: 'Date and time when this access authorization stops being valid unless it is extended.',
  lastAccess: 'Most recent recorded successful use of this access record.',
  auditTime: 'Timestamp when the audited access event was recorded by the API.',
  auditEvent: 'Access-control event recorded by the API, such as claim, grant, login, denial, session replacement or administrative update.',
  auditUser: 'Friendly user name associated with the audited access event when available.',
  auditIdentity: 'Verified Google identity associated with the access event when available.',
  auditRole: 'Permission profile associated with the access event.',
  auditResult: 'Whether the access-control event completed successfully or was denied.',
  auditClient: 'Client IP address recorded for the access-control event.',
}
