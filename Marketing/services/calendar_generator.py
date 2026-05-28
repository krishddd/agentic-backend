"""
Calendar Generator - Creates ICS files in-memory
"""
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

from ics import Calendar, Event


class CalendarGenerator:
    """Generates ICS calendar invites from scheduling text"""
    
    def parse_datetime(self, text: str) -> Optional[datetime]:
        """Parse datetime from natural language text"""
        text_lower = text.lower()
        now = datetime.now()
        
        day_names = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        # Try ISO format
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})\s*(?:at\s*)?(\d{1,2}:\d{2})?', text)
        if iso_match:
            date_str = iso_match.group(1)
            time_str = iso_match.group(2) or "10:00"
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        
        # Parse time
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text_lower)
        hour, minute = 10, 0
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            meridiem = time_match.group(3)
            if meridiem == 'pm' and hour < 12:
                hour += 12
            elif meridiem == 'am' and hour == 12:
                hour = 0
        
        target_date = None
        
        # Check day names
        for day_name, day_num in day_names.items():
            if day_name in text_lower:
                days_ahead = day_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                if 'next' in text_lower:
                    days_ahead += 7
                target_date = now + timedelta(days=days_ahead)
                break
        
        if 'tomorrow' in text_lower:
            target_date = now + timedelta(days=1)
        
        # Check month + day
        month_match = re.search(
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2})',
            text_lower
        )
        if month_match:
            month_abbr = month_match.group(1)[:3]
            day = int(month_match.group(2))
            months = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            month = months.get(month_abbr, now.month)
            year = now.year if month >= now.month else now.year + 1
            target_date = datetime(year, month, day)
        
        if target_date is None:
            target_date = now + timedelta(days=1)
            while target_date.weekday() >= 5:
                target_date += timedelta(days=1)
        
        return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    def generate_ics(
        self,
        scheduling_text: str,
        client_name: str,
        company_name: str,
        call_type: str,
        duration_minutes: int = 60
    ) -> Optional[bytes]:
        """Generate ICS calendar invite"""
        if not scheduling_text or len(scheduling_text.strip()) < 3:
            return None
        
        meeting_time = self.parse_datetime(scheduling_text)
        if not meeting_time:
            return None
        
        cal = Calendar()
        event = Event()
        
        event.name = f"{call_type} - {company_name} ({client_name})"
        event.begin = meeting_time
        event.duration = timedelta(minutes=duration_minutes)
        event.description = f"Follow-up {call_type.lower()} with {client_name} from {company_name}."
        
        cal.events.add(event)
        
        print(f"[Calendar] ✓ Generated invite for {meeting_time.strftime('%Y-%m-%d %H:%M')}")
        return str(cal).encode('utf-8')


# Singleton instance
_calendar_generator = None

def get_calendar_generator() -> CalendarGenerator:
    """Get or create calendar generator singleton"""
    global _calendar_generator
    if _calendar_generator is None:
        _calendar_generator = CalendarGenerator()
    return _calendar_generator
